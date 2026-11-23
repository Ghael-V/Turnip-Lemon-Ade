# Copyright 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
# SPDX-License-Identifier: MIT

"""Hand-written ctypes bindings for the Gallium ML interface.

check_abi() compares every offset, size and enum value with the ones libtorx
was built with.
"""

import ctypes
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    Union,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_uint,
    c_ubyte,
    c_void_p,
)

_libtorx = ctypes.CDLL("libtorx.so")

class pipe_tensor(Structure):
    pass

pipe_tensor._fields_ = [
    ("data",        POINTER(c_ubyte)),
    ("index",       c_uint),
    ("dims",        c_uint * 4),
    ("rank",        c_uint),
    ("scale",       c_float),
    ("scales",      POINTER(c_float)),
    ("zero_point",  c_int),
    ("zero_points", POINTER(c_int)),
    ("is_signed",   c_bool),
    ("is_constant", c_bool),
    ("is_external_output", c_bool),
    ("type_size",   c_ubyte),
]

PIPE_ML_OPERATION_TYPE_ADD             = 0
PIPE_ML_OPERATION_TYPE_CONVOLUTION     = 1
PIPE_ML_OPERATION_TYPE_POOLING         = 2
PIPE_ML_OPERATION_TYPE_CONCATENATION   = 3
PIPE_ML_OPERATION_TYPE_PACK            = 4
PIPE_ML_OPERATION_TYPE_SPLIT           = 5
PIPE_ML_OPERATION_TYPE_UNPACK          = 6
PIPE_ML_OPERATION_TYPE_PAD             = 7
PIPE_ML_OPERATION_TYPE_SCATTER_ND      = 8
PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED = 9
PIPE_ML_OPERATION_TYPE_BATCH_MATMUL    = 10
PIPE_ML_OPERATION_TYPE_RESHAPE         = 11
PIPE_ML_OPERATION_TYPE_RELU            = 12
PIPE_ML_OPERATION_TYPE_ABSOLUTE        = 13
PIPE_ML_OPERATION_TYPE_LOGISTIC        = 14
PIPE_ML_OPERATION_TYPE_TANH            = 15
PIPE_ML_OPERATION_TYPE_RSQRT           = 16
PIPE_ML_OPERATION_TYPE_HSWISH          = 17
PIPE_ML_OPERATION_TYPE_SUBTRACT        = 18
PIPE_ML_OPERATION_TYPE_TRANSPOSE       = 19
PIPE_ML_OPERATION_TYPE_STRIDED_SLICE   = 20
PIPE_ML_OPERATION_TYPE_RESIZE          = 21
PIPE_ML_OPERATION_TYPE_RESIZE_BILINEAR = 22
PIPE_ML_OPERATION_TYPE_ARGMAX          = 23
PIPE_ML_OPERATION_TYPE_SPACE_TO_BATCH  = 24
PIPE_ML_OPERATION_TYPE_BATCH_TO_SPACE  = 25
PIPE_ML_OPERATION_TYPE_MUL             = 26
PIPE_ML_OPERATION_TYPE_LEAKY_RELU      = 27
PIPE_ML_OPERATION_TYPE_QUANTIZE        = 28
PIPE_ML_OPERATION_TYPE_MAXIMUM         = 29
PIPE_ML_OPERATION_TYPE_MINIMUM         = 30
PIPE_ML_OPERATION_TYPE_SOFTMAX         = 31
PIPE_ML_OPERATION_TYPE_MEAN            = 32

PIPE_ML_POOLING_TYPE_AVG = 0
PIPE_ML_POOLING_TYPE_MAX = 1

class _conv(Structure):
    _fields_ = [
        ("weight_tensor",          POINTER(pipe_tensor)),
        ("bias_tensor",            POINTER(pipe_tensor)),
        ("stride_x",               c_uint),
        ("stride_y",               c_uint),
        ("padding_top",            c_uint),
        ("padding_bottom",         c_uint),
        ("padding_left",           c_uint),
        ("padding_right",          c_uint),
        ("pointwise",              c_bool),
        ("depthwise",              c_bool),
        ("activation_min",         c_int),
        ("activation_max",         c_int),
        ("relu",                   c_bool),
        ("dilation_width_factor",  c_uint),
        ("dilation_height_factor", c_uint),
    ]

class _pooling(Structure):
    _fields_ = [
        ("type",           c_int),
        ("stride_x",       c_uint),
        ("stride_y",       c_uint),
        ("filter_width",   c_uint),
        ("filter_height",  c_uint),
        ("padding_top",    c_uint),
        ("padding_bottom", c_uint),
        ("padding_left",   c_uint),
        ("padding_right",  c_uint),
    ]

class _pad(Structure):
    _fields_ = [
        ("before_x", c_uint),
        ("after_x",  c_uint),
        ("before_y", c_uint),
        ("after_y",  c_uint),
        ("before_z", c_uint),
        ("after_z",  c_uint),
        ("raw_zero", c_bool),
    ]

class _fcon(Structure):
    _fields_ = [
        ("weight_tensor", POINTER(pipe_tensor)),
        ("bias_tensor",   POINTER(pipe_tensor)),
        ("relu",          c_bool),
    ]

class _add(Structure):
    _fields_ = [("relu", c_bool)]

class _batch_matmul(Structure):
    _fields_ = [("adj_x", c_bool), ("adj_y", c_bool)]

class _conc(Structure):
    _fields_ = [("axis", c_int)]

class _split(Structure):
    _fields_ = [("axis", c_int)]

class _reshape(Structure):
    _fields_ = [("shape", c_uint * 4)]

class _transpose(Structure):
    _fields_ = [("perm", c_uint * 4)]

class _resize_bilinear(Structure):
    _fields_ = [("align_corners", c_bool), ("half_pixel_centers", c_bool)]

class _argmax(Structure):
    _fields_ = [("axis", c_int)]

class _space_batch(Structure):
    _fields_ = [
        ("block_y",  c_int),
        ("block_x",  c_int),
        ("before_y", c_int),
        ("after_y",  c_int),
        ("before_x", c_int),
        ("after_x",  c_int),
    ]

class _slice(Structure):
    _fields_ = [
        ("begin",   c_int * 4),
        ("end",     c_int * 4),
        ("strides", c_int * 4),
    ]

class _leakyrelu(Structure):
    _fields_ = [("alpha", c_float)]

class _softmax(Structure):
    _fields_ = [("beta", c_float)]

class _mean(Structure):
    _fields_ = [("axes", c_uint)]

class _op_union(Union):
    _fields_ = [
        ("conv",            _conv),
        ("pooling",         _pooling),
        ("pad",             _pad),
        ("fcon",            _fcon),
        ("add",             _add),
        ("batch_matmul",    _batch_matmul),
        ("conc",            _conc),
        ("split",           _split),
        ("reshape",         _reshape),
        ("transpose",       _transpose),
        ("resize_bilinear", _resize_bilinear),
        ("argmax",          _argmax),
        ("space_batch",     _space_batch),
        ("slice",           _slice),
        ("leakyrelu",       _leakyrelu),
        ("softmax",         _softmax),
        ("mean",            _mean),
    ]

class pipe_ml_operation(Structure):
    _anonymous_ = ("_u",)
    _fields_ = [
        ("type",           c_int),
        ("input_tensors",  POINTER(POINTER(pipe_tensor))),
        ("input_count",    c_uint),
        ("output_tensors", POINTER(POINTER(pipe_tensor))),
        ("output_count",   c_uint),
        ("_u",             _op_union),
    ]

class pipe_ml_device(Structure):
    pass

class pipe_ml_subgraph(Structure):
    pass

pipe_ml_subgraph._fields_ = [
    ("device", POINTER(pipe_ml_device)),
]

# Function pointer return values that are pointers use c_void_p so that
# callers can cast them (matches ctypesgen's UNCHECKED() wrapper behaviour).
pipe_ml_device._fields_ = [
    ("id",                    c_char_p),
    ("ml_operation_supported", CFUNCTYPE(c_bool,
        POINTER(pipe_ml_device), POINTER(pipe_ml_operation))),
    ("ml_subgraph_create",    CFUNCTYPE(c_void_p,
        POINTER(pipe_ml_device), POINTER(pipe_ml_operation), c_uint)),
    ("ml_subgraph_serialize", CFUNCTYPE(c_void_p,
        POINTER(pipe_ml_device), POINTER(pipe_ml_subgraph), POINTER(c_size_t))),
    ("ml_subgraph_destroy",   CFUNCTYPE(c_void_p,
        POINTER(pipe_ml_device), POINTER(pipe_ml_subgraph))),
    ("ml_device_destroy",     CFUNCTYPE(c_void_p,
        POINTER(pipe_ml_device))),
]

torx_device_probe = _libtorx.torx_device_probe
torx_device_probe.argtypes = []
torx_device_probe.restype = POINTER(pipe_ml_device)


# ---------------------------------------------------------------------------
# ABI self-check against torx_abi_layout() in torx_device.c
# ---------------------------------------------------------------------------

_torx_abi_layout = _libtorx.torx_abi_layout
_torx_abi_layout.argtypes = [POINTER(c_uint)]
_torx_abi_layout.restype = POINTER(ctypes.c_uint32)


def check_abi():
    """Compare this mirror's layout with the one libtorx was built with.

    The entries here and in torx_abi_layout() must stay in the same
    order; a count or value mismatch means the mirror is out of sync
    with p_state.h.
    """
    expected = [
        ("sizeof(pipe_tensor)", ctypes.sizeof(pipe_tensor), True),
        ("pipe_tensor.data", pipe_tensor.data.offset, True),
        ("pipe_tensor.index", pipe_tensor.index.offset, True),
        ("pipe_tensor.dims", pipe_tensor.dims.offset, True),
        ("pipe_tensor.rank", pipe_tensor.rank.offset, True),
        ("pipe_tensor.scale", pipe_tensor.scale.offset, True),
        ("pipe_tensor.scales", pipe_tensor.scales.offset, True),
        ("pipe_tensor.zero_point", pipe_tensor.zero_point.offset, True),
        ("pipe_tensor.zero_points", pipe_tensor.zero_points.offset, True),
        ("pipe_tensor.is_signed", pipe_tensor.is_signed.offset, True),
        ("pipe_tensor.is_constant", pipe_tensor.is_constant.offset, True),
        ("pipe_tensor.is_external_output",
         pipe_tensor.is_external_output.offset, True),
        ("pipe_tensor.type_size", pipe_tensor.type_size.offset, True),
        ("sizeof(pipe_ml_operation)", ctypes.sizeof(pipe_ml_operation), True),
        ("pipe_ml_operation.type", pipe_ml_operation.type.offset, True),
        ("pipe_ml_operation.input_tensors",
         pipe_ml_operation.input_tensors.offset, True),
        ("pipe_ml_operation.input_count",
         pipe_ml_operation.input_count.offset, True),
        ("pipe_ml_operation.output_tensors",
         pipe_ml_operation.output_tensors.offset, True),
        ("pipe_ml_operation.output_count",
         pipe_ml_operation.output_count.offset, True),
        ("pipe_ml_operation.conv.weight_tensor",
         pipe_ml_operation.conv.offset + _conv.weight_tensor.offset, True),
        ("pipe_ml_operation.conv.bias_tensor",
         pipe_ml_operation.conv.offset + _conv.bias_tensor.offset, True),
        ("pipe_ml_operation.conv.stride_x",
         pipe_ml_operation.conv.offset + _conv.stride_x.offset, True),
        ("pipe_ml_operation.conv.stride_y",
         pipe_ml_operation.conv.offset + _conv.stride_y.offset, True),
        ("pipe_ml_operation.conv.padding_top",
         pipe_ml_operation.conv.offset + _conv.padding_top.offset, True),
        ("pipe_ml_operation.conv.padding_bottom",
         pipe_ml_operation.conv.offset + _conv.padding_bottom.offset, True),
        ("pipe_ml_operation.conv.padding_left",
         pipe_ml_operation.conv.offset + _conv.padding_left.offset, True),
        ("pipe_ml_operation.conv.padding_right",
         pipe_ml_operation.conv.offset + _conv.padding_right.offset, True),
        ("pipe_ml_operation.conv.pointwise",
         pipe_ml_operation.conv.offset + _conv.pointwise.offset, True),
        ("pipe_ml_operation.conv.depthwise",
         pipe_ml_operation.conv.offset + _conv.depthwise.offset, True),
        ("pipe_ml_operation.conv.activation_min",
         pipe_ml_operation.conv.offset + _conv.activation_min.offset, True),
        ("pipe_ml_operation.conv.activation_max",
         pipe_ml_operation.conv.offset + _conv.activation_max.offset, True),
        ("pipe_ml_operation.conv.relu",
         pipe_ml_operation.conv.offset + _conv.relu.offset, True),
        ("pipe_ml_operation.conv.dilation_width_factor",
         pipe_ml_operation.conv.offset + _conv.dilation_width_factor.offset,
         True),
        ("pipe_ml_operation.conv.dilation_height_factor",
         pipe_ml_operation.conv.offset + _conv.dilation_height_factor.offset,
         True),
        ("pipe_ml_operation.pooling.type",
         pipe_ml_operation.pooling.offset + _pooling.type.offset, True),
        ("pipe_ml_operation.pooling.stride_x",
         pipe_ml_operation.pooling.offset + _pooling.stride_x.offset, True),
        ("pipe_ml_operation.pooling.stride_y",
         pipe_ml_operation.pooling.offset + _pooling.stride_y.offset, True),
        ("pipe_ml_operation.pooling.filter_width",
         pipe_ml_operation.pooling.offset + _pooling.filter_width.offset, True),
        ("pipe_ml_operation.pooling.filter_height",
         pipe_ml_operation.pooling.offset + _pooling.filter_height.offset, True),
        ("pipe_ml_operation.pooling.padding_top",
         pipe_ml_operation.pooling.offset + _pooling.padding_top.offset, True),
        ("pipe_ml_operation.pooling.padding_bottom",
         pipe_ml_operation.pooling.offset + _pooling.padding_bottom.offset, True),
        ("pipe_ml_operation.pooling.padding_left",
         pipe_ml_operation.pooling.offset + _pooling.padding_left.offset, True),
        ("pipe_ml_operation.pooling.padding_right",
         pipe_ml_operation.pooling.offset + _pooling.padding_right.offset, True),
        ("pipe_ml_operation.pad.before_x",
         pipe_ml_operation.pad.offset + _pad.before_x.offset, True),
        ("pipe_ml_operation.pad.after_x",
         pipe_ml_operation.pad.offset + _pad.after_x.offset, True),
        ("pipe_ml_operation.pad.before_y",
         pipe_ml_operation.pad.offset + _pad.before_y.offset, True),
        ("pipe_ml_operation.pad.after_y",
         pipe_ml_operation.pad.offset + _pad.after_y.offset, True),
        ("pipe_ml_operation.pad.before_z",
         pipe_ml_operation.pad.offset + _pad.before_z.offset, True),
        ("pipe_ml_operation.pad.after_z",
         pipe_ml_operation.pad.offset + _pad.after_z.offset, True),
        ("pipe_ml_operation.pad.raw_zero",
         pipe_ml_operation.pad.offset + _pad.raw_zero.offset, True),
        ("pipe_ml_operation.fcon.weight_tensor",
         pipe_ml_operation.fcon.offset + _fcon.weight_tensor.offset, True),
        ("pipe_ml_operation.fcon.bias_tensor",
         pipe_ml_operation.fcon.offset + _fcon.bias_tensor.offset, True),
        ("pipe_ml_operation.fcon.relu",
         pipe_ml_operation.fcon.offset + _fcon.relu.offset, True),
        ("pipe_ml_operation.add.relu",
         pipe_ml_operation.add.offset + _add.relu.offset, True),
        ("pipe_ml_operation.batch_matmul.adj_x",
         pipe_ml_operation.batch_matmul.offset + _batch_matmul.adj_x.offset, True),
        ("pipe_ml_operation.batch_matmul.adj_y",
         pipe_ml_operation.batch_matmul.offset + _batch_matmul.adj_y.offset, True),
        ("pipe_ml_operation.conc.axis",
         pipe_ml_operation.conc.offset + _conc.axis.offset, True),
        ("pipe_ml_operation.split.axis",
         pipe_ml_operation.split.offset + _split.axis.offset, True),
        ("pipe_ml_operation.reshape.shape",
         pipe_ml_operation.reshape.offset + _reshape.shape.offset, True),
        ("pipe_ml_operation.transpose.perm",
         pipe_ml_operation.transpose.offset + _transpose.perm.offset, True),
        ("pipe_ml_operation.resize_bilinear.align_corners",
         pipe_ml_operation.resize_bilinear.offset +
         _resize_bilinear.align_corners.offset, True),
        ("pipe_ml_operation.resize_bilinear.half_pixel_centers",
         pipe_ml_operation.resize_bilinear.offset +
         _resize_bilinear.half_pixel_centers.offset, True),
        ("pipe_ml_operation.argmax.axis",
         pipe_ml_operation.argmax.offset + _argmax.axis.offset, True),
        ("pipe_ml_operation.space_batch.block_y",
         pipe_ml_operation.space_batch.offset + _space_batch.block_y.offset, True),
        ("pipe_ml_operation.space_batch.block_x",
         pipe_ml_operation.space_batch.offset + _space_batch.block_x.offset, True),
        ("pipe_ml_operation.space_batch.before_y",
         pipe_ml_operation.space_batch.offset + _space_batch.before_y.offset, True),
        ("pipe_ml_operation.space_batch.after_y",
         pipe_ml_operation.space_batch.offset + _space_batch.after_y.offset, True),
        ("pipe_ml_operation.space_batch.before_x",
         pipe_ml_operation.space_batch.offset + _space_batch.before_x.offset, True),
        ("pipe_ml_operation.space_batch.after_x",
         pipe_ml_operation.space_batch.offset + _space_batch.after_x.offset, True),
        ("pipe_ml_operation.slice.begin",
         pipe_ml_operation.slice.offset + _slice.begin.offset, True),
        ("pipe_ml_operation.slice.end",
         pipe_ml_operation.slice.offset + _slice.end.offset, True),
        ("pipe_ml_operation.slice.strides",
         pipe_ml_operation.slice.offset + _slice.strides.offset, True),
        ("pipe_ml_operation.leakyrelu.alpha",
         pipe_ml_operation.leakyrelu.offset + _leakyrelu.alpha.offset, True),
        ("pipe_ml_operation.softmax.beta",
         pipe_ml_operation.softmax.offset + _softmax.beta.offset, True),
        ("pipe_ml_operation.mean.axes",
         pipe_ml_operation.mean.offset + _mean.axes.offset, True),
        ("sizeof(pipe_ml_subgraph)", ctypes.sizeof(pipe_ml_subgraph), True),
        ("pipe_ml_subgraph.device", pipe_ml_subgraph.device.offset, True),
        ("sizeof(pipe_ml_device)", ctypes.sizeof(pipe_ml_device), True),
        ("pipe_ml_device.id", pipe_ml_device.id.offset, True),
        ("pipe_ml_device.ml_operation_supported",
         pipe_ml_device.ml_operation_supported.offset, True),
        ("pipe_ml_device.ml_subgraph_create",
         pipe_ml_device.ml_subgraph_create.offset, True),
        ("pipe_ml_device.ml_subgraph_serialize",
         pipe_ml_device.ml_subgraph_serialize.offset, True),
        ("pipe_ml_device.ml_subgraph_destroy",
         pipe_ml_device.ml_subgraph_destroy.offset, True),
        ("pipe_ml_device.ml_device_destroy",
         pipe_ml_device.ml_device_destroy.offset, True),
        ("PIPE_ML_OPERATION_TYPE_ADD", PIPE_ML_OPERATION_TYPE_ADD, True),
        ("PIPE_ML_OPERATION_TYPE_CONVOLUTION",
         PIPE_ML_OPERATION_TYPE_CONVOLUTION, True),
        ("PIPE_ML_OPERATION_TYPE_POOLING",
         PIPE_ML_OPERATION_TYPE_POOLING, True),
        ("PIPE_ML_OPERATION_TYPE_CONCATENATION",
         PIPE_ML_OPERATION_TYPE_CONCATENATION, True),
        ("PIPE_ML_OPERATION_TYPE_PACK", PIPE_ML_OPERATION_TYPE_PACK, True),
        ("PIPE_ML_OPERATION_TYPE_SPLIT", PIPE_ML_OPERATION_TYPE_SPLIT, True),
        ("PIPE_ML_OPERATION_TYPE_UNPACK", PIPE_ML_OPERATION_TYPE_UNPACK, True),
        ("PIPE_ML_OPERATION_TYPE_PAD", PIPE_ML_OPERATION_TYPE_PAD, True),
        ("PIPE_ML_OPERATION_TYPE_SCATTER_ND",
         PIPE_ML_OPERATION_TYPE_SCATTER_ND, True),
        ("PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED",
         PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED, True),
        ("PIPE_ML_OPERATION_TYPE_BATCH_MATMUL",
         PIPE_ML_OPERATION_TYPE_BATCH_MATMUL, True),
        ("PIPE_ML_OPERATION_TYPE_RESHAPE",
         PIPE_ML_OPERATION_TYPE_RESHAPE, True),
        ("PIPE_ML_OPERATION_TYPE_RELU", PIPE_ML_OPERATION_TYPE_RELU, True),
        ("PIPE_ML_OPERATION_TYPE_ABSOLUTE",
         PIPE_ML_OPERATION_TYPE_ABSOLUTE, True),
        ("PIPE_ML_OPERATION_TYPE_LOGISTIC",
         PIPE_ML_OPERATION_TYPE_LOGISTIC, True),
        ("PIPE_ML_OPERATION_TYPE_TANH", PIPE_ML_OPERATION_TYPE_TANH, True),
        ("PIPE_ML_OPERATION_TYPE_RSQRT", PIPE_ML_OPERATION_TYPE_RSQRT, True),
        ("PIPE_ML_OPERATION_TYPE_HSWISH", PIPE_ML_OPERATION_TYPE_HSWISH, True),
        ("PIPE_ML_OPERATION_TYPE_SUBTRACT",
         PIPE_ML_OPERATION_TYPE_SUBTRACT, True),
        ("PIPE_ML_OPERATION_TYPE_TRANSPOSE",
         PIPE_ML_OPERATION_TYPE_TRANSPOSE, True),
        ("PIPE_ML_OPERATION_TYPE_STRIDED_SLICE",
         PIPE_ML_OPERATION_TYPE_STRIDED_SLICE, True),
        ("PIPE_ML_OPERATION_TYPE_RESIZE", PIPE_ML_OPERATION_TYPE_RESIZE, True),
        ("PIPE_ML_OPERATION_TYPE_RESIZE_BILINEAR",
         PIPE_ML_OPERATION_TYPE_RESIZE_BILINEAR, True),
        ("PIPE_ML_OPERATION_TYPE_ARGMAX", PIPE_ML_OPERATION_TYPE_ARGMAX, True),
        ("PIPE_ML_OPERATION_TYPE_SPACE_TO_BATCH",
         PIPE_ML_OPERATION_TYPE_SPACE_TO_BATCH, True),
        ("PIPE_ML_OPERATION_TYPE_BATCH_TO_SPACE",
         PIPE_ML_OPERATION_TYPE_BATCH_TO_SPACE, True),
        ("PIPE_ML_OPERATION_TYPE_MUL", PIPE_ML_OPERATION_TYPE_MUL, True),
        ("PIPE_ML_OPERATION_TYPE_LEAKY_RELU",
         PIPE_ML_OPERATION_TYPE_LEAKY_RELU, True),
        ("PIPE_ML_OPERATION_TYPE_QUANTIZE",
         PIPE_ML_OPERATION_TYPE_QUANTIZE, True),
        ("PIPE_ML_OPERATION_TYPE_MAXIMUM",
         PIPE_ML_OPERATION_TYPE_MAXIMUM, True),
        ("PIPE_ML_OPERATION_TYPE_MINIMUM",
         PIPE_ML_OPERATION_TYPE_MINIMUM, True),
        ("PIPE_ML_OPERATION_TYPE_SOFTMAX",
         PIPE_ML_OPERATION_TYPE_SOFTMAX, True),
        ("PIPE_ML_OPERATION_TYPE_MEAN", PIPE_ML_OPERATION_TYPE_MEAN, True),
        ("PIPE_ML_POOLING_TYPE_AVG", PIPE_ML_POOLING_TYPE_AVG, True),
        ("PIPE_ML_POOLING_TYPE_MAX", PIPE_ML_POOLING_TYPE_MAX, True),
    ]

    count = c_uint()
    values = _torx_abi_layout(ctypes.byref(count))
    if count.value != len(expected):
        raise RuntimeError(
            f"Torx: gallium.py expects {len(expected)} ABI entries, libtorx "
            f"reports {count.value}; the ctypes mirror is out of sync with "
            f"p_state.h")
    for i, (name, want, exact) in enumerate(expected):
        got = values[i]
        if (got != want) if exact else (got < want):
            raise RuntimeError(
                f"Torx: ABI mismatch for {name}: gallium.py has {want}, "
                f"libtorx reports {got}; the ctypes mirror is out of sync "
                f"with p_state.h")

