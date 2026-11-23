# Copyright © 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# SPDX-License-Identifier: MIT

import ctypes
import logging
import operator
from dataclasses import dataclass, field

import torch
from executorch.exir.backend.utils import exir_ops

from . import gallium
from .dim_order_pass import delegates_layout_op, dim_order, is_data_layout_op
from .fold_qdq_with_annotated_qparams_pass import QuantArgs

supported_ops = {
    exir_ops.edge.aten.convolution.default: gallium.PIPE_ML_OPERATION_TYPE_CONVOLUTION,
    exir_ops.edge.aten.add.Tensor: gallium.PIPE_ML_OPERATION_TYPE_ADD,
    exir_ops.edge.aten.cat.default: gallium.PIPE_ML_OPERATION_TYPE_CONCATENATION,
    exir_ops.edge.aten.sigmoid.default: gallium.PIPE_ML_OPERATION_TYPE_LOGISTIC,
    exir_ops.edge.aten.upsample_nearest2d.vec: gallium.PIPE_ML_OPERATION_TYPE_RESIZE,
    exir_ops.edge.aten.view_copy.default: gallium.PIPE_ML_OPERATION_TYPE_RESHAPE,
    exir_ops.edge.aten.permute_copy.default: gallium.PIPE_ML_OPERATION_TYPE_RESHAPE,
    exir_ops.edge.aten.max_pool2d_with_indices.default: gallium.PIPE_ML_OPERATION_TYPE_POOLING,
}

POOLING_STRIDE_MAX = 3

CHANNEL_DIM = 1
SPATIAL_DIMS = (2, 3)

# Activation ops that are fused into the preceding convolution or add.
# Two variants: FUSABLE_ACTIVATION_OPS uses ExIR edge-dialect op references
# (used during lowering and partitioning); FUSABLE_ACTIVATION_OPS_ATEN uses
# base-aten op references (used during PT2E quantizer annotation).
FUSABLE_ACTIVATION_OPS = {
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.relu_.default,
    exir_ops.edge.aten.hardtanh.default,
    exir_ops.edge.aten.hardtanh_.default,
    exir_ops.edge.aten.clamp.default,
    exir_ops.edge.aten.clamp_.default,
}
FUSABLE_ACTIVATION_OPS_ATEN = {
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
    torch.ops.aten.hardtanh.default,
    torch.ops.aten.hardtanh_.default,
    torch.ops.aten.clamp.default,
    torch.ops.aten.clamp_.default,
}
RELU_OPS = {
    exir_ops.edge.aten.relu.default,
    exir_ops.edge.aten.relu_.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
}


def activation_is_absorbable(activation: torch.fx.Node,
                             producer_is_conv: bool) -> bool:
    """Return whether producer's operation can absorb this activation.

    Convolutions carry arbitrary clamp bounds in activation_min/max; add
    only has a relu flag, so its clamp must start at zero and end at 6 or
    be unbounded above.
    """
    if activation.target in RELU_OPS:
        return True
    if producer_is_conv:
        return True
    low = activation.args[1] if len(activation.args) > 1 else None
    high = activation.args[2] if len(activation.args) > 2 else None
    return low == 0.0 and (high is None or high == 6.0)


ACTIVATION_ABSORBING_OPS = {
    gallium.PIPE_ML_OPERATION_TYPE_CONVOLUTION,
    gallium.PIPE_ML_OPERATION_TYPE_ADD,
}


def fused_activation(node: torch.fx.Node) -> torch.fx.Node | None:
    """Return the activation user absorbed by node's operation, if any."""
    if node.target not in supported_ops:
        return None
    op_type = supported_ops[node.target]
    if op_type not in ACTIVATION_ABSORBING_OPS:
        return None
    users = list(node.users.keys())
    if len(users) != 1 or users[0].target not in FUSABLE_ACTIVATION_OPS:
        return None
    is_conv = (op_type == gallium.PIPE_ML_OPERATION_TYPE_CONVOLUTION)
    return users[0] if activation_is_absorbable(users[0], is_conv) else None


# Identity quantization parameters — a mathematical no-op (scale=1.0, zp=0).
# Used as placeholder when FoldAndAnnotateQParamsPass hasn't run yet
# (e.g. during partitioning), so operation structs carry valid scale/zp
# fields for ml_operation_supported().
IDENTITY_QPARAMS = QuantArgs(
    scale=1.0, zp=0, qmin=-128, qmax=127, dtype=torch.int8
)

@dataclass
class GalliumGraph:
    """Converted Gallium graph plus owned backing storage for ctypes pointers."""

    operations: list[gallium.pipe_ml_operation] = field(default_factory=list)
    node_to_tensor_idx: dict[torch.fx.Node, int] = field(default_factory=dict)
    next_tensor_id: int = -1
    _buffers: list[object] = field(default_factory=list, repr=False, compare=False)

    def own(self, *buffers: object):
        """Keep driver-visible storage alive until release_buffers()."""
        self._buffers.extend(buffers)

    def release_buffers(self):
        """Free owned storage once the driver holds its own copies."""
        self._buffers.clear()

    def allocate_tensor_id(self) -> int:
        self.next_tensor_id += 1
        return self.next_tensor_id

    def resolve_idx(self, node: torch.fx.Node, default: int | None = None) -> int | None:
        """Return Gallium tensor index for a node, or default if unknown."""
        gidx = self.node_to_tensor_idx.get(node)
        return gidx if gidx is not None else default

def _require_quant_args_dtype_in(
    qparams: QuantArgs, role: str, allowed_dtypes: tuple[torch.dtype, ...]
) -> None:
    if qparams.dtype not in allowed_dtypes:
        allowed = ", ".join([str(dtype) for dtype in allowed_dtypes])
        raise ValueError(
            f"Torx currently supports quantized int8 models only. "
            f"Expected {role} quantization dtype(s) {allowed}, got {qparams.dtype}"
        )

def _derive_bias_qparams(input_qparams: QuantArgs, weight_qparams: QuantArgs) -> QuantArgs:
    if input_qparams.per_channel:
        raise ValueError("Per-channel activation quantization is not supported for bias derivation")

    qmin = torch.iinfo(torch.int32).min
    qmax = torch.iinfo(torch.int32).max

    if weight_qparams.per_channel:
        input_scale = input_qparams.get_scale_per_tensor()
        weight_scales = weight_qparams.get_scale_per_channel()
        scales = [input_scale * weight_scale for weight_scale in weight_scales]
        return QuantArgs(
            scale=scales,
            zp=[0 for _ in scales],
            qmin=qmin,
            qmax=qmax,
            dtype=torch.int32,
            axis=weight_qparams.axis,
            per_channel=True,
        )

    scale = input_qparams.get_scale_per_tensor() * weight_qparams.get_scale_per_tensor()
    return QuantArgs(
        scale=scale,
        zp=0,
        qmin=qmin,
        qmax=qmax,
        dtype=torch.int32,
        axis=0,
        per_channel=False,
    )

def fill_tensor(tensor: torch._subclasses.fake_tensor.FakeTensor, ptensor: gallium.pipe_tensor, graph: GalliumGraph, qparams: QuantArgs | None = None, *, index: int | None = None, is_constant: bool = False, dim_order: tuple):
    if index is not None:
        ptensor.index = index
    else:
        ptensor.index = graph.allocate_tensor_id()

    ptensor.rank = tensor.ndim
    ptensor.is_constant = is_constant

    if qparams is not None and qparams.per_channel:
        scales = (ctypes.c_float * len(qparams.scale))(*qparams.scale)
        zero_points = (ctypes.c_int * len(qparams.zp))(*qparams.zp)
        ptensor.scales = ctypes.cast(scales, ctypes.POINTER(ctypes.c_float))
        ptensor.zero_points = ctypes.cast(zero_points, ctypes.POINTER(ctypes.c_int))
        ptensor.scale = 0.0
        ptensor.zero_point = 0
        graph.own(scales, zero_points)
    elif qparams is not None:
        ptensor.scale = qparams.scale
        ptensor.zero_point = qparams.zp

    if qparams is not None and qparams.dtype in (torch.qint8, torch.qint32, torch.int8, torch.int32):
        ptensor.is_signed = True
    elif qparams is not None and qparams.dtype == torch.quint8:
        ptensor.is_signed = False
    else:
        ptensor.is_signed = tensor.dtype == torch.qint8

    _dtype_to_size = {
        torch.int8: 1, torch.qint8: 1, torch.quint8: 1,
        torch.int32: 4, torch.qint32: 4,
        torch.float32: 4,
    }
    if qparams is not None and qparams.dtype in _dtype_to_size:
        ptensor.type_size = _dtype_to_size[qparams.dtype]
    elif tensor.dtype in _dtype_to_size:
        ptensor.type_size = _dtype_to_size[tensor.dtype]
    else:
        ptensor.type_size = tensor.element_size()

    storage = [tensor.shape[axis] for axis in dim_order]
    assert len(storage) <= 4
    ptensor.dims[:] = [1] * (4 - len(storage)) + storage

def _set_tensor_data(graph: GalliumGraph, ptensor: gallium.pipe_tensor, data: torch.Tensor):
    """Point ptensor.data at the tensor's storage, which the graph owns."""
    graph.own(data)
    ptensor.data = ctypes.cast(ctypes.c_void_p(data.data_ptr()),
                               ctypes.POINTER(ctypes.c_uint8))


def fill_conv(node: torch.fx.Node, poperation: gallium.pipe_ml_operation, graph: GalliumGraph, input_qparams: QuantArgs, weight_qparams: QuantArgs, bias_qparams: QuantArgs):
    weights = node.args[1]

    # Detect depthwise early — needed for weight layout decisions below.
    groups = node.args[8]
    input_channels = node.args[0].meta['val'].shape[1]  # NCHW in ExIR
    is_depthwise = (groups == input_channels and input_channels > 1)

    # PyTorch stores convolution weights [C_out, C_in/g, kH, kW].  Gallium ML
    # expects a full convolution's as [C_out, kH, kW, C_in] and a depthwise's as
    # [1, kH, kW, C_out].
    weight_order = (1, 2, 3, 0) if is_depthwise else (0, 2, 3, 1)

    poperation.conv.weight_tensor = ctypes.pointer(gallium.pipe_tensor())
    fill_tensor(weights.meta['val'], poperation.conv.weight_tensor.contents, graph, weight_qparams, is_constant=True, dim_order=weight_order)

    if 'torx_data' in weights.meta:
        contents = weights.meta['torx_data'].permute(*weight_order).contiguous()
        _set_tensor_data(graph, poperation.conv.weight_tensor.contents, contents)

    biases = node.args[2] if len(node.args) > 2 else None
    if biases is not None:
        poperation.conv.bias_tensor = ctypes.pointer(gallium.pipe_tensor())
        # The Ethos-U NPU expects biases quantized with scale = input_scale * weight_scale.
        # Use derived qparams (not the independently observed ones from PyTorch).
        driver_bias_qp = _derive_bias_qparams(input_qparams, weight_qparams)
        fill_tensor(biases.meta['val'], poperation.conv.bias_tensor.contents, graph, driver_bias_qp, is_constant=True, dim_order=tuple(range(biases.meta['val'].ndim)))
        if 'torx_data' in biases.meta:
            # Re-quantize bias from PyTorch's observed scale to the scale the NPU
            # expects (input_scale * weight_scale).
            bias_data = biases.meta['torx_data']
            src_zp = torch.tensor(bias_qparams.zp, dtype=torch.float64)
            src_scale = torch.tensor(bias_qparams.scale, dtype=torch.float64)
            bias_float = (bias_data.to(torch.float64) - src_zp) * src_scale
            if driver_bias_qp.per_channel:
                scales = torch.tensor(driver_bias_qp.scale, dtype=torch.float64)
                bias_requantized = torch.round(bias_float / scales).clamp(
                    torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max
                ).to(torch.int32).contiguous()
            else:
                bias_requantized = torch.round(bias_float / driver_bias_qp.scale).clamp(
                    torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max
                ).to(torch.int32).contiguous()
            _set_tensor_data(graph, poperation.conv.bias_tensor.contents,
                             bias_requantized)
    else:
        poperation.conv.bias_tensor = None

    # aten.convolution.default args layout:
    #   0: input, 1: weight, 2: bias, 3: stride, 4: padding,
    #   5: dilation, 6: transposed, 7: output_padding, 8: groups

    # Stride — args[3] is [stride_h, stride_w]
    stride = node.args[3]
    poperation.conv.stride_y = stride[0]
    poperation.conv.stride_x = stride[1]

    # Dilation — args[5] is [dilation_h, dilation_w]
    dilation = node.args[5]
    poperation.conv.dilation_height_factor = dilation[0]
    poperation.conv.dilation_width_factor = dilation[1]

    # Padding — args[4] is [pad_h, pad_w] (symmetric in PyTorch).
    #
    # PyTorch specifies symmetric padding: pad_h is added on BOTH top and
    # bottom, pad_w on BOTH left and right.  However the Ethos-U NPU needs
    # the actual per-side padding that each output position will use.
    #
    # With stride > 1 the last kernel window may not reach the right/bottom
    # padding at all, so we derive pad_right/pad_bottom from the geometry
    # instead of blindly mirroring the PyTorch value.  pad_left/pad_top
    # are kept as-is because they determine the kernel alignment (the first
    # output position starts at -pad_left/-pad_top).
    padding = node.args[4]
    poperation.conv.padding_top = padding[0]
    poperation.conv.padding_left = padding[1]

    input_shape = node.args[0].meta['val'].shape   # NCHW
    output_shape = node.meta['val'].shape           # NCHW
    weight_shape = weights.meta['val'].shape        # [O, I/g, kH, kW]

    eff_kernel_h = (weight_shape[2] - 1) * dilation[0] + 1
    eff_kernel_w = (weight_shape[3] - 1) * dilation[1] + 1

    poperation.conv.padding_bottom = max(
        0, (output_shape[2] - 1) * stride[0] - padding[0] + eff_kernel_h - input_shape[2]
    )
    poperation.conv.padding_right = max(
        0, (output_shape[3] - 1) * stride[1] - padding[1] + eff_kernel_w - input_shape[3]
    )

    poperation.conv.depthwise = is_depthwise

    # Pointwise — 1×1 kernel (weight shape is [out_ch, in_ch/groups, kH, kW] in ExIR)
    weight_shape = weights.meta['val'].shape
    poperation.conv.pointwise = (weight_shape[2] == 1 and weight_shape[3] == 1)

    # The relu flag only describes a true relu; bounded activations are
    # carried by activation_min/max instead.
    users = list(node.users.keys())
    poperation.conv.relu = (
        len(users) == 1 and users[0].target in (exir_ops.edge.aten.relu.default,
                                                exir_ops.edge.aten.relu_.default)
    )


def _fused_activation_bounds(node: torch.fx.Node) -> tuple[float | None, float | None]:
    activation = fused_activation(node)
    if activation is None:
        return None, None

    if activation.target in RELU_OPS:
        return 0.0, None
    low = activation.args[1] if len(activation.args) > 1 else None
    high = activation.args[2] if len(activation.args) > 2 else None
    return low, high


def set_conv_activation_range(node: torch.fx.Node,
                              poperation: gallium.pipe_ml_operation,
                              output_qp: QuantArgs):
    scale = output_qp.get_scale_per_tensor()
    zp = output_qp.zp

    poperation.conv.activation_min = output_qp.qmin
    poperation.conv.activation_max = output_qp.qmax

    low, high = _fused_activation_bounds(node)
    # int(x + 0.5) matches teflon's C rounding, not Python's round().
    if low is not None:
        poperation.conv.activation_min = max(output_qp.qmin,
                                             zp + int(low / scale + 0.5))
    if high is not None:
        poperation.conv.activation_max = min(output_qp.qmax,
                                             zp + int(high / scale + 0.5))


def is_2x_resize(node: torch.fx.Node) -> bool:
    """Return whether a resize node doubles both spatial dimensions.

    Shapes are in the node's own NCHW order, so the spatial dimensions are
    2 and 3 here; fill_tensor moves them to 1 and 2 when it writes NHWC.
    """
    input_node = node.args[0]
    if not isinstance(input_node, torch.fx.Node):
        return False

    input_val = input_node.meta.get('val')
    output_val = node.meta.get('val')
    if not isinstance(input_val, torch.Tensor) or input_val.dim() != 4:
        return False
    if not isinstance(output_val, torch.Tensor) or output_val.dim() != 4:
        return False

    return all(output_val.shape[dim] == 2 * input_val.shape[dim]
               for dim in SPATIAL_DIMS)


def _spatial_pair(value, default: tuple[int, int]) -> tuple[int, int]:
    """Normalise an aten int-or-list spatial argument to (height, width)."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return (value[0], value[-1])
    return (value, value)


def _max_pool_geometry(node: torch.fx.Node):
    """Return max pooling's (kernel, stride, padding) as (height, width) pairs."""
    kernel = _spatial_pair(node.args[1], (1, 1))
    stride = _spatial_pair(node.args[2] if len(node.args) > 2 else None, kernel)
    padding = _spatial_pair(node.args[3] if len(node.args) > 3 else None, (0, 0))
    return kernel, stride, padding


def max_pool_values(node: torch.fx.Node) -> torch.fx.Node | None:
    """Return the getitem carrying max pooling's values output, or None when
    the node is not a max pooling this frontend can lower.

    aten.max_pool2d decomposes to a two-output node whose second output is
    the argmax indices; the Gallium pooling member expresses neither those
    indices nor dilation nor ceil mode.
    """
    if node.target != exir_ops.edge.aten.max_pool2d_with_indices.default:
        return None

    _, stride, _ = _max_pool_geometry(node)
    if not all(1 <= s <= POOLING_STRIDE_MAX for s in stride):
        return None
    if _spatial_pair(node.args[4] if len(node.args) > 4 else None, (1, 1)) != (1, 1):
        return None
    if len(node.args) > 5 and node.args[5]:
        return None

    users = list(node.users.keys())
    if len(users) != 1:
        return None
    values = users[0]
    if values.target is not operator.getitem or values.args[1] != 0:
        return None
    return values


def fill_pooling(node: torch.fx.Node, output_node: torch.fx.Node,
                 poperation: gallium.pipe_ml_operation):
    kernel, stride, padding = _max_pool_geometry(node)

    poperation.pooling.type = gallium.PIPE_ML_POOLING_TYPE_MAX
    poperation.pooling.filter_height = kernel[0]
    poperation.pooling.filter_width = kernel[1]
    poperation.pooling.stride_y = stride[0]
    poperation.pooling.stride_x = stride[1]
    poperation.pooling.padding_top = padding[0]
    poperation.pooling.padding_left = padding[1]

    # PyTorch pads symmetrically; the NPU needs the padding each output
    # position actually uses, so derive the trailing sides from the geometry.
    input_shape = node.args[0].meta['val'].shape   # NCHW
    output_shape = output_node.meta['val'].shape   # NCHW

    poperation.pooling.padding_bottom = max(
        0, (output_shape[2] - 1) * stride[0] - padding[0] + kernel[0] - input_shape[2]
    )
    poperation.pooling.padding_right = max(
        0, (output_shape[3] - 1) * stride[1] - padding[1] + kernel[1] - input_shape[3]
    )


def exir_to_operation(node: torch.fx.Node, graph: GalliumGraph) -> gallium.pipe_ml_operation | None:
    if node.target not in supported_ops:
        return None

    # Max pooling's tensor output is carried by a getitem, so that node — not
    # this one — holds the output shape and the quantisation parameters.
    output_node = node
    if node.target == exir_ops.edge.aten.max_pool2d_with_indices.default:
        output_node = max_pool_values(node)
        if output_node is None:
            return None

    # Qparams are populated by FoldAndAnnotateQParamsPass during preprocess().
    # During partitioning the pass hasn't run yet, so we fall back to identity
    # quantization (scale=1.0, zp=0) — enough for ml_operation_supported().
    input_qparams = node.meta.get("input_qparams", {})
    output_qparams = output_node.meta.get("output_qparams", {})

    # When a fusable activation follows this node, FoldAndAnnotateQParamsPass
    # places output_qparams on the *activation* node (since Q follows it, not
    # the conv).  Propagate those qparams here so the conv's output tensor
    # gets the correct scale/zp for the NPU.
    if not output_qparams:
        activation = fused_activation(node)
        if activation is not None:
            output_qparams = activation.meta.get("output_qparams", {})

    output_val = output_node.meta.get('val')
    if not isinstance(output_val, torch.Tensor):
        return None
    rank = output_val.dim()

    operation = gallium.pipe_ml_operation()
    operation.type = supported_ops[node.target]

    # Rank 3 is admitted only for the view, permute and concatenation
    # the dim order pass admitted; every other operation reaches the
    # driver as rank-4 NHWC, and a rank-4 view or permute is not lowered.
    layout_op = is_data_layout_op(node)
    if layout_op:
        if not delegates_layout_op(node):
            return None
    elif rank != 4 or operation.type == gallium.PIPE_ML_OPERATION_TYPE_RESHAPE:
        return None

    if (operation.type == gallium.PIPE_ML_OPERATION_TYPE_RESIZE and
            not is_2x_resize(node)):
        return None

    if operation.type == gallium.PIPE_ML_OPERATION_TYPE_CONCATENATION:
        if not isinstance(node.args[0], (list, tuple)):
            return None
        dim = (node.args[1] if len(node.args) > 1 else 0) % rank
        if rank == 4 and dim != CHANNEL_DIM:
            return None
        input_nodes = list(node.args[0])
        list_arg_qparams = input_qparams.get(0, IDENTITY_QPARAMS)
        input_qps = [list_arg_qparams] * len(input_nodes)
    else:
        if operation.type == gallium.PIPE_ML_OPERATION_TYPE_ADD:
            input_nodes = list(node.args[:2])
        else:
            input_nodes = [node.args[0]]
        input_qps = [input_qparams.get(i, IDENTITY_QPARAMS)
                     for i in range(len(input_nodes))]

    operation.input_count = len(input_nodes)
    operation.input_tensors = (ctypes.POINTER(gallium.pipe_tensor) * operation.input_count)()

    for inp_idx, input_node in enumerate(input_nodes):
        if not isinstance(input_node, torch.fx.Node):
            return None
        input_tensor = input_node.meta['val']
        if input_tensor.dim() not in ((3, 4) if layout_op else (4,)):
            return None
        input_qp = input_qps[inp_idx]
        _require_quant_args_dtype_in(input_qp, f"input{inp_idx}", (torch.qint8, torch.int8))
        operation.input_tensors[inp_idx] = ctypes.pointer(gallium.pipe_tensor())
        existing_idx = graph.node_to_tensor_idx.get(input_node)
        fill_tensor(input_tensor, operation.input_tensors[inp_idx].contents, graph, input_qp, index=existing_idx, dim_order=dim_order(input_node))
        if input_node not in graph.node_to_tensor_idx:
            graph.node_to_tensor_idx[input_node] = operation.input_tensors[inp_idx].contents.index

    operation.output_count = 1
    operation.output_tensors = (ctypes.POINTER(gallium.pipe_tensor) * operation.output_count)()

    output_tensor = output_val
    output_qp = output_qparams.get(0, IDENTITY_QPARAMS)
    _require_quant_args_dtype_in(output_qp, "output", (torch.qint8, torch.int8))
    operation.output_tensors[0] = ctypes.pointer(gallium.pipe_tensor())
    fill_tensor(output_tensor, operation.output_tensors[0].contents, graph, output_qp, dim_order=dim_order(output_node))
    graph.node_to_tensor_idx[output_node] = operation.output_tensors[0].contents.index

    if operation.type == gallium.PIPE_ML_OPERATION_TYPE_CONVOLUTION:
        weight_qp = input_qparams.get(1, IDENTITY_QPARAMS)
        _require_quant_args_dtype_in(weight_qp, "weights", (torch.qint8, torch.int8))

        # Bias is optional — node.args[2] can be None for Conv2d(bias=False)
        has_bias = len(node.args) > 2 and node.args[2] is not None
        if has_bias:
            bias_qp = input_qparams.get(2)
            if bias_qp is None:
                bias_qp = _derive_bias_qparams(input_qp, weight_qp)
            _require_quant_args_dtype_in(bias_qp, "bias", (torch.qint32, torch.int32))
        else:
            bias_qp = None

        fill_conv(node, operation, graph, input_qp, weight_qp, bias_qp)
        set_conv_activation_range(node, operation, output_qp)

    elif operation.type == gallium.PIPE_ML_OPERATION_TYPE_ADD:
        operation.add.relu = fused_activation(node) is not None

    elif operation.type == gallium.PIPE_ML_OPERATION_TYPE_CONCATENATION:
        operation.conc.axis = 4 - rank + dim_order(output_node).index(dim)

    elif operation.type == gallium.PIPE_ML_OPERATION_TYPE_RESHAPE:
        out = operation.output_tensors[0].contents
        for i in range(4):
            operation.reshape.shape[i] = out.dims[i]

    elif operation.type == gallium.PIPE_ML_OPERATION_TYPE_POOLING:
        fill_pooling(node, output_node, operation)

    return operation


def is_torx_convertible(node: torch.fx.Node) -> bool:
    """Return whether a node can be lowered to a Gallium ML operation."""
    return exir_to_operation(node, GalliumGraph()) is not None


def is_torx_supported(node: torch.fx.Node, ml_device) -> bool:
    """Return whether a node is both lowerable and accepted by the ML device."""
    graph = GalliumGraph()
    operation = exir_to_operation(node, graph)
    if operation is None:
        return False
    return ml_device.contents.ml_operation_supported(ml_device, operation)

def exir_to_gallium(nodes: list[torch.fx.Node]) -> GalliumGraph:
    """Translate supported ExIR nodes into a GalliumGraph.

    The returned object owns both translated operations and backing storage
    required by ctypes pointers embedded in those operations.
    """
    graph = GalliumGraph()
    for n in nodes:
        if n.target in FUSABLE_ACTIVATION_OPS:
            producer = n.args[0]
            if (isinstance(producer, torch.fx.Node) and
                    fused_activation(producer) is n):
                continue
            raise RuntimeError(
                f"Torx: activation '{n.name}' cannot be absorbed by its "
                f"producer and has no standalone lowering")
        if n.target is operator.getitem:
            producer = n.args[0]
            if (isinstance(producer, torch.fx.Node) and
                    max_pool_values(producer) is n):
                continue
            raise RuntimeError(
                f"Torx: '{n.name}' does not select the values output of a "
                f"delegated max pooling and has no lowering")
        operation = exir_to_operation(n, graph)
        if operation is not None:
            if operation.type == gallium.PIPE_ML_OPERATION_TYPE_CONVOLUTION:
                if not operation.conv.weight_tensor.contents.data:
                    raise RuntimeError(f"Torx: convolution '{n.name}' has no weight data")
                if operation.conv.bias_tensor and not operation.conv.bias_tensor.contents.data:
                    raise RuntimeError(f"Torx: convolution '{n.name}' has no bias data")
            else:
                args = [a for arg in n.args
                        for a in (arg if isinstance(arg, (list, tuple)) else [arg])]
                for arg in args:
                    if isinstance(arg, torch.fx.Node) and 'torx_data' in arg.meta:
                        raise RuntimeError(
                            f"Torx: '{n.name}' reads constant tensor '{arg.name}', "
                            f"which only convolutions support")
            graph.operations.append(operation)
            # Alias a fused activation to the operation's output index so
            # subsequent consumers reuse the same tensor.
            activation = fused_activation(n)
            if activation is not None:
                graph.node_to_tensor_idx[activation] = operation.output_tensors[0].contents.index
        elif n.target in supported_ops:
            raise RuntimeError(
                f"Torx: '{n.name}' was delegated but could not be lowered")
    return graph

ml_device = None

def get_ml_device() -> gallium.pipe_ml_device | None:
    global ml_device

    if ml_device:
        return ml_device

    ml_device = gallium.torx_device_probe()

    if not ml_device:
        raise RuntimeError("ML device not found")

    logging.debug(f"ML device found: {ml_device.contents.id.decode()}")

    return ml_device
