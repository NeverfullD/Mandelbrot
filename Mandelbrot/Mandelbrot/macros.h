//       $Id: macros.h 41657 2020-11-12 08:32:22Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/pfc/inc/pfc/macros.h $
// $Revision: 41657 $
//     $Date: 2020-11-12 09:32:22 +0100 (Do., 12 Nov 2020) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: November 7, 2020
// Copyright: (c) 2020 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#pragma once

#include "./compiler_detection.h"

// -------------------------------------------------------------------------------------------------

#undef  PFC_MACRO_EXPAND
#define PFC_MACRO_EXPAND(x) \
   x

#undef  PFC_STATIC_ASSERT
#define PFC_STATIC_ASSERT(c) \
   static_assert ((c), PFC_STRINGIZE (c))

#undef  PFC_STRINGIZE
#define PFC_STRINGIZE(x) \
   #x

// -------------------------------------------------------------------------------------------------

#undef CATTR_DEVICE
#undef CATTR_GPU_ENABLED
#undef CATTR_GPU_ENABLED_INL
#undef CATTR_HOST
#undef CATTR_INLINE
#undef CATTR_RESTRICT

#if defined PFC_DETECTED_COMPILER_NVCC
   #define CATTR_DEVICE   __device__
   #define CATTR_HOST     __host__
   #define CATTR_INLINE   __forceinline__
   #define CATTR_RESTRICT __restrict__
#else
   #define CATTR_DEVICE
   #define CATTR_HOST
   #define CATTR_INLINE   inline
   #define CATTR_RESTRICT __restrict
#endif

#define CATTR_GPU_ENABLED        CATTR_HOST CATTR_DEVICE
#define CATTR_GPU_ENABLED_INLINE CATTR_HOST CATTR_DEVICE CATTR_INLINE
