# Copyright © 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# SPDX-License-Identifier: MIT

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from torch.fx.passes.infra.pass_base import PassBase, PassResult

from .fold_qdq_with_annotated_qparams_pass import is_dequantize, is_quantize

DIM_ORDER_KEY = "torx_dim_order"
LAYOUT_RUN_KEY = "torx_layout_run"


def contiguous_order(rank: int) -> tuple:
    return tuple(range(rank))


def is_contiguous(order: tuple) -> bool:
    return order == contiguous_order(len(order))


def is_identity_clone(node: torch.fx.Node) -> bool:
    """Return whether node is a _clone_dim_order that keeps the dim order."""
    if node.target != exir_ops.edge.dim_order_ops._clone_dim_order.default:
        return False
    val = node.meta.get("val")
    input_val = node.args[0].meta.get("val")
    return (val is not None and input_val is not None and
            val.dim_order() == input_val.dim_order())


def dim_order(node: torch.fx.Node) -> tuple:
    node = _producer(node)
    order = node.meta.get(DIM_ORDER_KEY)
    if order is None:
        raise RuntimeError(
            f"Torx: '{node.name}' carries no dim order; AnnotateDimOrderPass "
            f"has not run on its graph")
    if node.op == "placeholder" and len(order) < 4 and not is_contiguous(order):
        raise RuntimeError(
            f"Torx: '{node.name}' enters the delegate in storage order "
            f"{order}, but a tensor below rank 4 crossing the boundary is contiguous")
    return order


def layout_run(node: torch.fx.Node) -> frozenset | None:
    return node.meta.get(LAYOUT_RUN_KEY)


def delegates_layout_op(node: torch.fx.Node) -> bool:
    return layout_run(node) is not None


def _rank(node) -> int | None:
    val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    return val.dim() if isinstance(val, torch.Tensor) else None


def _is_transparent(node: torch.fx.Node) -> bool:
    return is_quantize(node) or is_dequantize(node) or is_identity_clone(node)


def _producer(node):
    while isinstance(node, torch.fx.Node) and _is_transparent(node):
        node = node.args[0]
    return node


def _consumers(node: torch.fx.Node):
    for user in node.users:
        if _is_transparent(user):
            yield from _consumers(user)
        else:
            yield user


def _inputs(node: torch.fx.Node) -> list:
    if node.target == exir_ops.edge.aten.cat.default:
        return list(node.args[0]) if isinstance(node.args[0], (list, tuple)) else []
    return [node.args[0]]


def _quantisation_unchanged(node: torch.fx.Node) -> bool:
    quant = next((user for user in node.users if is_quantize(user)), None)
    if quant is None:
        return True
    for inp in _inputs(node):
        if not is_dequantize(inp):
            return False
        if (inp.args[1], inp.args[2]) != (quant.args[1], quant.args[2]):
            return False
    return True


def is_data_layout_op(node) -> bool:
    return (isinstance(node, torch.fx.Node) and node.op == "call_function" and
            node.target in (exir_ops.edge.aten.view_copy.default,
                            exir_ops.edge.aten.permute_copy.default,
                            exir_ops.edge.aten.cat.default) and
            _rank(node) == 3)


def merges_spatial_dims(node: torch.fx.Node) -> bool:
    in_val = node.args[0].meta.get("val") if isinstance(node.args[0], torch.fx.Node) else None
    out_val = node.meta.get("val")
    if not isinstance(in_val, torch.Tensor) or in_val.dim() != 4:
        return False
    if not isinstance(out_val, torch.Tensor) or out_val.dim() != 3:
        return False
    return (out_val.shape[0] == in_val.shape[0] and
            out_val.shape[1] == in_val.shape[1] and
            out_val.shape[2] == in_val.shape[2] * in_val.shape[3])


def _order(node: torch.fx.Node, orders: dict) -> tuple | None:
    inputs = _inputs(node)
    if not inputs:
        return None
    input_orders = []
    for inp in inputs:
        producer = _producer(inp)
        if producer in orders:
            if orders[producer] is None:
                return None
            input_orders.append(orders[producer])
        elif DIM_ORDER_KEY in producer.meta:
            input_orders.append(producer.meta[DIM_ORDER_KEY])
        else:
            return None
    if node.target == exir_ops.edge.aten.view_copy.default:
        return (0, 2, 1) if merges_spatial_dims(node) else None
    if any(_rank(inp) != 3 for inp in inputs):
        return None
    if node.target == exir_ops.edge.aten.permute_copy.default:
        perm = node.args[1]
        inverse = [0] * len(perm)
        for i, p in enumerate(perm):
            inverse[p] = i
        return tuple(inverse[d] for d in input_orders[0])
    if any(order != input_orders[0] for order in input_orders):
        return None
    return input_orders[0]


def _run(node: torch.fx.Node) -> set:
    run = set()
    pending = [node]
    while pending:
        n = pending.pop()
        if n in run:
            continue
        run.add(n)
        for inp in _inputs(n):
            producer = _producer(inp)
            if is_data_layout_op(producer):
                pending.append(producer)
        for user in _consumers(n):
            if is_data_layout_op(user):
                pending.append(user)
    return run


class AnnotateDimOrderPass(PassBase):

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            node.meta.pop(DIM_ORDER_KEY, None)
            node.meta.pop(LAYOUT_RUN_KEY, None)

        # A rank-4 tensor is stored NHWC; one below rank 4 keeps its logical
        # order and sits in the trailing entries of the four dims.
        for node in graph_module.graph.nodes:
            rank = _rank(node)
            if rank is not None:
                node.meta[DIM_ORDER_KEY] = (0, 2, 3, 1) if rank == 4 else contiguous_order(rank)

        orders = {}
        for node in graph_module.graph.nodes:
            if is_data_layout_op(node):
                orders[node] = (_order(node, orders)
                                if _quantisation_unchanged(node) else None)

        decided = set()
        for node in orders:
            if node in decided:
                continue
            run = _run(node)
            decided |= run
            if any(orders[n] is None for n in run):
                continue
            if any(not is_contiguous(orders[n]) and
                   any(c not in run for c in _consumers(n)) for n in run):
                continue
            members = frozenset(n.name for n in run)
            for n in run:
                n.meta[DIM_ORDER_KEY] = orders[n]
                n.meta[LAYOUT_RUN_KEY] = members

        return PassResult(graph_module, False)
