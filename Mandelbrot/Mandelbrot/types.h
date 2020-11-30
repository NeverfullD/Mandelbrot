//       $Id: types.h 41657 2020-11-12 08:32:22Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/pfc/inc/pfc/types.h $
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

#include <cstddef>
#include <cstdint>

// -------------------------------------------------------------------------------------------------

namespace pfc {

using byte_t  = std::uint8_t;    // std::byte
using dword_t = std::uint32_t;   //
using long_t  = std::int32_t;    //
using word_t  = std::uint16_t;   //

}   // namespace pfc

PFC_STATIC_ASSERT (sizeof (pfc::byte_t)  == 1);
PFC_STATIC_ASSERT (sizeof (pfc::dword_t) == 4);
PFC_STATIC_ASSERT (sizeof (pfc::long_t)  == 4);
PFC_STATIC_ASSERT (sizeof (pfc::word_t)  == 2);

// -------------------------------------------------------------------------------------------------

namespace pfc {

#pragma pack (push, 1)
   struct BGR_3_t final {
      byte_t blue;
      byte_t green;
      byte_t red;
   };
#pragma pack (pop)

#if defined PFC_KNOW_PRAGMA_WARNING_PUSH_POP
#pragma warning (push)
#pragma warning (disable: 4201)   // nameless struct/union
#endif

#pragma pack (push, 1)
   struct BGR_4_t final {
      union {
         BGR_3_t bgr_3;

         struct {
            byte_t blue;
            byte_t green;
            byte_t red;
         };
      };

      byte_t unused;
   };
#pragma pack (pop)

#if defined PFC_KNOW_PRAGMA_WARNING_PUSH_POP
#pragma warning (pop)
#endif

using pixel_t      = BGR_4_t;
using pixel_file_t = BGR_3_t;

}   // namespace pfc

PFC_STATIC_ASSERT (sizeof (pfc::BGR_3_t) == 3);
PFC_STATIC_ASSERT (sizeof (pfc::BGR_4_t) == 4);
