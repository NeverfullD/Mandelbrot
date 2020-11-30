//       $Id: libraries.h 41657 2020-11-12 08:32:22Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/pfc/inc/pfc/libraries.h $
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

#include "./macros.h"

// -------------------------------------------------------------------------------------------------

#if defined PFC_DETECTED_COMPILER_NVCC
   #define PFC_DO_NOT_USE_GSL
   #define PFC_DO_NOT_USE_VLD
   #define PFC_DO_NOT_USE_WINDOWS
#endif

// -------------------------------------------------------------------------------------------------

#undef PFC_HAVE_VLD
#undef PFC_VLD_INCLUDED

#if __has_include (<vld.h>) && !defined PFC_DO_NOT_USE_VLD   // Visual Leak Detector (https://kinddragon.github.io/vld)
   #include <vld.h>

   #define PFC_HAVE_VLD
   #define PFC_VLD_INCLUDED

   #pragma message ("PFC: using 'Visual Leak Detector'")
#else
   #pragma message ("PFC: not using 'Visual Leak Detector'")
#endif

// -------------------------------------------------------------------------------------------------

#undef PFC_HAVE_GSL
#undef PFC_GSL_INCLUDED

#if __has_include (<gsl/gsl>) && !defined PFC_DO_NOT_USE_GSL   // Guideline Support Library (https://github.com/Microsoft/GSL)
   #include <gsl/gsl>

   #define PFC_HAVE_GSL
   #define PFC_GSL_INCLUDED

   #pragma message ("PFC: using 'Guideline Support Library'")
#else
   #pragma message ("PFC: not using 'Guideline Support Library'")
#endif

// -------------------------------------------------------------------------------------------------

#undef PFC_HAVE_WINDOWS
#undef PFC_WINDOWS_INCLUDED

#if __has_include (<windows.h>) && !defined PFC_DO_NOT_USE_WINDOWS
   #undef  NOMINMAX
   #define NOMINMAX

   #undef  STRICT
   #define STRICT

   #undef  VC_EXTRALEAN
   #define VC_EXTRALEAN

   #undef  WIN32_LEAN_AND_MEAN
   #define WIN32_LEAN_AND_MEAN

   #include <windows.h>

   #define PFC_HAVE_WINDOWS
   #define PFC_WINDOWS_INCLUDED

   #pragma message ("PFC: using 'windows.h'")
#else
   #pragma message ("PFC: not using 'windows.h'")
#endif
