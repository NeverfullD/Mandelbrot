//       $Id: compiler_detection.h 41681 2020-11-13 10:17:06Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/pfc/inc/pfc/compiler_detection.h $
// $Revision: 41681 $
//     $Date: 2020-11-13 11:17:06 +0100 (Fr., 13 Nov 2020) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: November 7, 2020
// Copyright: (c) 2020 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#pragma once

#include <type_traits>

// -------------------------------------------------------------------------------------------------

#undef PFC_DETECTED_COMPILER_CL
#undef PFC_DETECTED_COMPILER_CLANG
#undef PFC_DETECTED_COMPILER_GCC
#undef PFC_DETECTED_COMPILER_ICC
#undef PFC_DETECTED_COMPILER_NONE
#undef PFC_DETECTED_COMPILER_NVCC
#undef PFC_DETECTED_COMPILER_TYPE

#if defined __clang__
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_clang_t
   #define PFC_DETECTED_COMPILER_CLANG

   #pragma message ("PFC: LLVM C++ compiler Clang detected")

#elif defined __CUDACC__
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_nvcc_t
   #define PFC_DETECTED_COMPILER_NVCC

   #pragma message ("PFC: Nvidia C++ compiler nvcc detected")

#elif defined __GNUC__
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_gcc_t
   #define PFC_DETECTED_COMPILER_GCC

   #pragma message ("PFC: GNU C++ compiler g++ detected")

#elif defined __INTEL_COMPILER
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_icc_t
   #define PFC_DETECTED_COMPILER_ICC

   #pragma message ("PFC: Intel C++ compiler detected")

#elif defined _MSC_VER
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_cl_t
   #define PFC_DETECTED_COMPILER_CL

   #pragma message ("PFC: Microsoft C++ compiler detected")

#else
   #define PFC_DETECTED_COMPILER_TYPE pfc::detected_compiler_none_t
   #define PFC_DETECTED_COMPILER_NONE

   #pragma message ("PFC: unknown C++ compiler detected")
#endif

// -------------------------------------------------------------------------------------------------

#if defined __cplusplus
   #if (__cplusplus >= 199702L) && (__cplusplus <= 199799L)
      #pragma message ("PFC: compiler uses C++98 standard (maybe use cl-compiler option '/Zc:__cplusplus')")

   #elif (__cplusplus >= 201102L) && (__cplusplus <= 201199L)
      #pragma message ("PFC: compiler uses C++11 standard")

   #elif (__cplusplus >= 201402L) && (__cplusplus <= 201499L)
      #pragma message ("PFC: compiler uses C++14 standard")

   #elif (__cplusplus >= 201702L) && (__cplusplus <= 201799L)
      #pragma message ("PFC: compiler uses C++17 standard")

   #elif (__cplusplus >= 202002L) && (__cplusplus <= 202099L)
      #pragma message ("PFC: compiler uses C++20 standard")

   #else
      #pragma message ("PFC: compiler uses an unknown C++ standard (maybe use cl-compiler option '/Zc:__cplusplus')")
   #endif
#else
   #pragma message ("PFC: compiler seems not to be a C++ compiler")
#endif

// -------------------------------------------------------------------------------------------------

#undef PFC_KNOW_PRAGMA_WARNING_PUSH_POP

#if defined PFC_DETECTED_COMPILER_CL || defined PFC_DETECTED_COMPILER_NVCC
   #define PFC_KNOW_PRAGMA_WARNING_PUSH_POP
#endif

// -------------------------------------------------------------------------------------------------

namespace pfc {

enum class compiler {
   none, cl, clang, gcc, icc, nvcc
};

using detected_compiler_none_t  = std::integral_constant <compiler, compiler::none>;
using detected_compiler_cl_t    = std::integral_constant <compiler, compiler::cl>;
using detected_compiler_clang_t = std::integral_constant <compiler, compiler::clang>;
using detected_compiler_gcc_t   = std::integral_constant <compiler, compiler::gcc>;
using detected_compiler_icc_t   = std::integral_constant <compiler, compiler::icc>;
using detected_compiler_nvcc_t  = std::integral_constant <compiler, compiler::nvcc>;

using detected_compiler_t = PFC_DETECTED_COMPILER_TYPE;

constexpr auto detected_compiler_v {detected_compiler_t::value};

}   // namespace pfc
