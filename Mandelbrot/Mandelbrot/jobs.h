//       $Id: jobs.h 41402 2020-09-30 13:53:18Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/DSE/HPC3/2020-WS/ILV/src/jobs/src/jobs.h $
// $Revision: 41402 $
//     $Date: 2020-09-30 15:53:18 +0200 (Mi., 30 Sep 2020) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: September 29, 2020
// Copyright: (c) 2020 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#pragma once

#include <cassert>
#include <complex>
#include <concepts>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// -------------------------------------------------------------------------------------------------

template <std::floating_point T = float> class jobs final {
   public:
      using real_t    = T;
      using complex_t = std::complex <real_t>;
      using pair_t    = std::pair <real_t, real_t>;

      explicit jobs (std::string const & name) {
         read_jobs (std::ifstream {name});
      }

      complex_t get_center (std::size_t const i) const {
         return {
            at <2, comp::first > (i),
            at <2, comp::second> (i)
         };
      }

      complex_t get_lower_left (std::size_t const i) const {
         return {
            at <0, comp::first > (i), 
            at <0, comp::second> (i)
         };
      }

      auto const & get_ratio () const {
         return m_ratio;
      }

      complex_t get_size (std::size_t const i) const {
         return {
            at <3, comp::first > (i), 
            at <3, comp::second> (i)
         };
      }

      complex_t get_upper_right (std::size_t const i) const {
         return {
            at <1, comp::first > (i), 
            at <1, comp::second> (i)
         };
      }

      auto size () const {
         return std::size (m_jobs);
      }

   private:
      enum class comp {
         first, second
      };

      template <std::size_t I, comp C> real_t & at (std::size_t const i) {
         assert (i < std::size (m_jobs));

         if constexpr (C == comp::first ) return std::get <I> (m_jobs[i]).first;
         if constexpr (C == comp::second) return std::get <I> (m_jobs[i]).second;
      }

      template <std::size_t I, comp C> real_t const & at (std::size_t const i) const {
         assert (i < std::size (m_jobs));

         if constexpr (C == comp::first ) return std::get <I> (m_jobs[i]).first;
         if constexpr (C == comp::second) return std::get <I> (m_jobs[i]).second;
      }

      void read_jobs (std::istream && in) {
         if (in) {
            std::size_t images {}; m_ratio = 0; m_jobs.clear ();

            if ((in >> images >> m_ratio) && (images > 0) && (m_ratio > 0)) {
               m_jobs.resize (images);

               for (std::size_t i {}; i < images; ++i) {
                  char        c     {};
                  std::size_t image {};

                  in >> image >> c >> c
                     >> at <0, comp::first> (i) >> c >> at <0, comp::second> (i) >> c >> c >> c
                     >> at <1, comp::first> (i) >> c >> at <1, comp::second> (i) >> c >> c >> c
                     >> at <2, comp::first> (i) >> c >> at <2, comp::second> (i) >> c >> c >> c
                     >> at <3, comp::first> (i) >> c >> at <3, comp::second> (i) >> c;

                  assert (image == i);
               }
            }
         }
      }

      real_t m_ratio {};
      std::vector <std::tuple <
         pair_t,   // 0: lower left corner
         pair_t,   // 1: upper right corner
         pair_t,   // 2: center point
         pair_t    // 3: size (width and height)
      >> m_jobs;
};
