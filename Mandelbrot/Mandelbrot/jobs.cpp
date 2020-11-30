//       $Id: jobs.cpp 41402 2020-09-30 13:53:18Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/vocational/teaching/DSE/HPC3/2020-WS/ILV/src/jobs/src/jobs.cpp $
// $Revision: 41402 $
//     $Date: 2020-09-30 15:53:18 +0200 (Mi., 30 Sep 2020) $
//   $Author: p20068 $
//   Creator: Peter Kulczycki
//  Creation: September 26, 2020
// Copyright: (c) 2020 Peter Kulczycki (peter.kulczycki<AT>fh-hagenberg.at)
//   License: This document contains proprietary information belonging to
//            University of Applied Sciences Upper Austria, Campus Hagenberg.
//            It is distributed under the Boost Software License (see
//            https://www.boost.org/users/license.html).

#include "./jobs.h"

#include <gsl/gsl>

#include <cmath>
#include <iomanip>
#include <string_view>

// -------------------------------------------------------------------------------------------------

template <typename T> std::ostream & operator << (std::ostream & lhs, std::complex <T> const & rhs) {
   return lhs << '{' << rhs.real () << ',' << rhs.imag () << '}';
}

// -------------------------------------------------------------------------------------------------

using int_t     = std::size_t;
using real_t    = long double;
using complex_t = std::complex <real_t>;

// -------------------------------------------------------------------------------------------------

constexpr int_t  bmp_count {200};
constexpr real_t bmp_ratio {16.0 / 9.0};

constexpr real_t imag_width  {4};
constexpr real_t imag_height {imag_width / bmp_ratio};

constexpr complex_t imag_center_start {0, 0};
constexpr complex_t imag_center_zoom  {-0.745289981, 0.113075003};

constexpr real_t zoom_factor {0.95};

// -------------------------------------------------------------------------------------------------

void write_jobs (std::ostream & out, int_t const images) {
   if (out) {
      real_t const zoom {
         std::pow (std::pow (zoom_factor, real_t {bmp_count} - 1), real_t {1} / (images - 1))
      };

      out << images    << '\n'
          << bmp_ratio << '\n';

      complex_t imag_center {imag_center_start};
      complex_t imag_size   {imag_width, imag_height};

      complex_t const imag_center_step {(imag_center_zoom - imag_center) / real_t (images)};

      for (int_t b {0}; b < images; ++b) {
         complex_t const imag_ll {imag_center - imag_size / real_t {2}};
         complex_t const imag_ur {imag_ll     + imag_size};

         out << std::setw (3)
             << b           << ','     // image # (i.e. zoom step #)
             << imag_ll     << ','     // lower left corner of image
             << imag_ur     << ','     // upper right corner of image
             << imag_center << ','     // center point of image
             << imag_size   << '\n';   // size (width and height) of image

         imag_center += imag_center_step;
         imag_size   *= zoom;
      }
   }
}

void write_jobs (int_t const images) {
   std::ofstream out {
      "./jobs-" + std::string (std::max <std::size_t> (0, 3 - std::size (std::to_string (images))), '0') + std::to_string (images) + ".txt"
   };

   if (out) {
      out << std::fixed
          << std::setprecision (7);

      write_jobs (out, images);
   }
}

// -------------------------------------------------------------------------------------------------

int main () {
   write_jobs (  5);
   write_jobs ( 10);
   write_jobs ( 25);
   write_jobs ( 50);
   write_jobs (100);
   write_jobs (200);

   jobs <> const j {"./jobs-200.txt"};
   j.get_lower_left (0);
   j.size ();
}
