/*
 * Copyright © Microsoft Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#include "hmft_entrypoints.h"

#include "mfpipeinterop.h"

// utility to convert from pipe_h2645_enc_picture_type to eAVEncH264PictureType
// There is no eAVEncH265PictureType, so this is used for both
enum eAVEncH264PictureType
ConvertPictureTypeToAVEncH264PictureType( enum pipe_h2645_enc_picture_type picType )
{
   switch( picType )
   {
      case PIPE_H2645_ENC_PICTURE_TYPE_P:
         return eAVEncH264PictureType_P;
      case PIPE_H2645_ENC_PICTURE_TYPE_B:
         return eAVEncH264PictureType_B;
      case PIPE_H2645_ENC_PICTURE_TYPE_IDR:
      case PIPE_H2645_ENC_PICTURE_TYPE_I:
      default:
         return eAVEncH264PictureType_IDR;
   }
}

// utility to convert from FourCC to pipe_format
enum pipe_format
ConvertFourCCToPipeFormat( DWORD dwFourCC )
{
   if( dwFourCC == FOURCC_NV12 )
   {
      return PIPE_FORMAT_NV12;
   }
   else if( dwFourCC == FOURCC_P010 )
   {
      return PIPE_FORMAT_P010;
   }
   else if( dwFourCC == FOURCC_AYUV )
   {
      return PIPE_FORMAT_AYUV;
   }
   else if( dwFourCC == FOURCC_YUY2 )
   {
      return PIPE_FORMAT_YUYV;
   }
   else if( dwFourCC == FOURCC_Y210 )
   {
      return PIPE_FORMAT_Y210;
   }
   else if( dwFourCC == FOURCC_Y410 )
   {
      return PIPE_FORMAT_Y410;
   }
   else
   {
      UNREACHABLE( "Unsupported FourCC" );
      return PIPE_FORMAT_NONE;
   }
}

// utility to convert from pipe_format to image stride
UINT32
AdjustStrideForPipeFormatAndWidth( enum pipe_format pipeFormat, UINT32 width )
{
   UINT32 stride = 0;
   switch( pipeFormat )
   {
      case PIPE_FORMAT_NV12:
         stride = width;
         break;
      case PIPE_FORMAT_P010:
         stride = 2 * width;
         break;
      case PIPE_FORMAT_YUYV:
         stride = 2 * width;
         break;
      case PIPE_FORMAT_AYUV:
         stride = 4 * width;
         break;
      case PIPE_FORMAT_Y210:
         stride = 4 * width;
         break;
      case PIPE_FORMAT_Y410:
         stride = 4 * width;
         break;
   }
   return stride;
}

// utility to convert from pipe_format to chroma format idc
UINT32
GetChromaFormatIdc( enum pipe_format pipeFormat )
{
   switch( pipeFormat )
   {
      case PIPE_FORMAT_NV12:
      case PIPE_FORMAT_P010:
         return 1;
      case PIPE_FORMAT_YUYV:
      case PIPE_FORMAT_Y210:
         return 2;
      case PIPE_FORMAT_AYUV:
      case PIPE_FORMAT_Y410:
         return 3;
      default:
      {
         UNREACHABLE( "Unsupported pipe video format" );
      }
      break;
   }
}

// utility to convert from errno to HRESULT
HRESULT
ConvertErrnoRetToHR( int ret )
{
   switch( ret )
   {
      case 0:
         return S_OK;
      case ENOMEM:
         return MF_E_INSUFFICIENT_BUFFER;
      case EINVAL:
         return E_INVALIDARG;
      default:
         return E_FAIL;
   }
}

