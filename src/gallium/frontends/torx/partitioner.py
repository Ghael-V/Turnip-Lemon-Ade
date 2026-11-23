# Copyright © 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# SPDX-License-Identifier: MIT

import operator

import torch

from executorch.exir.backend.backend_details import ExportedProgram
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import generate_pattern_op_partitions
from executorch.exir.backend.partitioner import (
   DelegationSpec,
   Partitioner,
   PartitionResult,
)
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.operator_support import OperatorSupportBase

from .backend import TorxBackend
from .debug import dump_graph_support, node_tensor_dtype
from .dim_order_pass import AnnotateDimOrderPass, is_identity_clone, layout_run
from .exir_to_gallium import (
    FUSABLE_ACTIVATION_OPS,
    fused_activation,
    get_ml_device,
    is_torx_convertible,
    is_torx_supported,
    max_pool_values,
)
from .fold_qdq_with_annotated_qparams_pass import is_dequantize, is_quantize


def _is_partitioned(node: torch.fx.Node, tag: str) -> bool:
    return node.meta.get("delegation_tag") == tag


PER_CHANNEL_QDQ_OPS = (
    exir_ops.edge.quantized_decomposed.quantize_per_channel.default,
    torch.ops.quantized_decomposed.quantize_per_channel.default,
    exir_ops.edge.quantized_decomposed.dequantize_per_channel.default,
    torch.ops.quantized_decomposed.dequantize_per_channel.default,
)

class TorxSupportedOperators(OperatorSupportBase):

    @staticmethod
    def _is_graph_input(node: torch.fx.Node) -> bool:
        return node.op == "placeholder" and "from_node" not in node.meta

    @staticmethod
    def _is_input_quantization(node: torch.fx.Node) -> bool:
        """A quantize_per_tensor that converts a graph input from float32 to int8."""
        if not is_quantize(node):
            return False
        return (
            len(node.args) > 0
            and isinstance(node.args[0], torch.fx.Node)
            and TorxSupportedOperators._is_graph_input(node.args[0])
        )

    @staticmethod
    def _is_output_dequantization(node: torch.fx.Node) -> bool:
        """A dequantize_per_tensor that converts int8 back to float32 for the graph output."""
        if not is_dequantize(node):
            return False
        users = list(node.users.keys())
        return len(users) == 1 and users[0].op == "output"

    @staticmethod
    def _is_frozen_param(node: torch.fx.Node) -> bool:
        """A placeholder carrying constant weights or biases (not a user input)."""
        return node.op == "placeholder" and "from_node" in node.meta

    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:

        if self._is_graph_input(node):
            return False

        if node.op == "output":
            return False

        if self._is_input_quantization(node):
            return False

        if self._is_output_dequantization(node):
            return False

        # Frozen parameter placeholders (weights, biases) belong in the delegate.
        if self._is_frozen_param(node):
            return True

        # Internal Q/DQ nodes (for weights, biases, inter-op) belong in the
        # delegate so FoldAndAnnotateQParamsPass can fold them in preprocess().
        if is_quantize(node) or is_dequantize(node):
            return True

        # Same for per-channel Q/DQ nodes, which are only used for weights and biases.
        if node.target in PER_CHANNEL_QDQ_OPS:
            return all(self.is_node_supported(submodules, user)
                       for user in node.users)

        # An activation is only delegated when its producer's operation
        # absorbs it; a standalone activation has no lowering.
        if node.target in FUSABLE_ACTIVATION_OPS:
            producer = node.args[0]
            return (isinstance(producer, torch.fx.Node) and
                    fused_activation(producer) is node)

        # Max pooling's values getitem carries the operation's output tensor,
        # so it is delegated together with the pooling that produces it.
        if node.target is operator.getitem:
            producer = node.args[0]
            return (isinstance(producer, torch.fx.Node) and
                    max_pool_values(producer) is node)

        # A clone fed directly by a graph input could end up passing that
        # input straight to the partition output once removed, leaving the
        # delegate's IO entries unresolvable.
        if is_identity_clone(node) and node.args[0].op != "placeholder":
            return True

        ml_device = get_ml_device()
        return is_torx_supported(node, ml_device)

class TorxPartitioner(Partitioner):
    def __init__(self) -> None:
        super().__init__({})
        self.op_support = TorxSupportedOperators()
        self.delegation_spec = DelegationSpec(
            TorxBackend.__name__,
            [],
        )

    def _detag_boundary_nodes(self, graph_module: torch.fx.GraphModule, tag: str) -> None:
        # Detag boundary Q/DQ just like Arm integer-only partitioning does:
        # keep Q/DQ on the host side and delegate only int kernels.
        for node in graph_module.graph.nodes:
            if not _is_partitioned(node, tag):
                continue

            is_q_node = is_quantize(node)
            is_dq_node = is_dequantize(node)

            is_boundary_q_node = (
                is_q_node
                and len(node.all_input_nodes) > 0
                and not _is_partitioned(node.all_input_nodes[0], tag)
            )
            is_boundary_dq_node = is_dq_node and any(
                not _is_partitioned(user, tag) for user in node.users
            )

            if is_boundary_q_node or is_boundary_dq_node:
                node.meta.pop("delegation_tag", None)
                continue

            if not is_q_node and not is_dq_node:
                for input_node in node.all_input_nodes:
                    if _is_partitioned(input_node, tag):
                        continue
                    dtype = node_tensor_dtype(input_node)
                    if dtype is not None and dtype != torch.int8:
                        node.meta.pop("delegation_tag", None)
                        break

    def partition(self, edge_exported_program: ExportedProgram) -> PartitionResult:
        AnnotateDimOrderPass()(edge_exported_program.graph_module)
        dump_graph_support(edge_exported_program.graph_module, self.op_support)

        partition_tags = {}
        partition_list = generate_pattern_op_partitions(
            edge_exported_program.graph_module, op_support=self.op_support
        )
        for partition in partition_list:
            delegation_tag = f"tag{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = delegation_tag

            self._detag_boundary_nodes(edge_exported_program.graph_module, delegation_tag)

            tagged_nodes = [
                n for n in partition.nodes
                if n.meta.get("delegation_tag") == delegation_tag
            ]
            has_ml_op = any(
                n.op == "call_function" and is_torx_convertible(n)
                for n in tagged_nodes
            )

            if not tagged_nodes or not has_ml_op:
                for n in tagged_nodes:
                    n.meta.pop("delegation_tag", None)
                continue

            partition_tags[delegation_tag] = self.delegation_spec

        graph_module = edge_exported_program.graph_module
        name_to_node = {n.name: n for n in graph_module.graph.nodes}
        for node in graph_module.graph.nodes:
            run = layout_run(node)
            if run is None:
                continue
            tags = {name_to_node[name].meta.get("delegation_tag")
                    for name in run if name in name_to_node}
            if len(tags) > 1:
                raise RuntimeError(
                    f"Torx: the data layout run {sorted(run)} was split "
                    f"across partitions; it must be delegated whole or not "
                    f"at all")

        return PartitionResult(
            tagged_exported_program=edge_exported_program,
            partition_tags=partition_tags,
        )
