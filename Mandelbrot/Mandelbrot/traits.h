//       $Id: traits.h 41681 2020-11-13 10:17:06Z p20068 $
//      $URL: https://svn01.fh-hagenberg.at/bin/cepheiden/pfc/trunk/pfc/inc/pfc/traits.h $
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

#include <cfloat>
#include <climits>
#include <iterator>
#include <limits>
#include <ratio>
#include <string>
#include <string_view>
#include <type_traits>

//namespace pfc {

// -------------------------------------------------------------------------------------------------

//template <typename T> constexpr bool is_integral_signed_v {
//   std::is_same_v <T, char> || std::is_same_v <T, short> || std::is_same_v <T, int> || std::is_same_v <T, long> || std::is_same_v <T, long long>
//};
//
//template <typename T> constexpr bool is_integral_unsigned_v {
//   std::is_same_v <T, unsigned char> || std::is_same_v <T, unsigned short> || std::is_same_v <T, unsigned> || std::is_same_v <T, unsigned long> || std::is_same_v <T, unsigned long long>
//};
//
//template <typename T> constexpr bool is_integral_v {
//   pfc::is_integral_signed_v <T> || pfc::is_integral_unsigned_v <T>
//};

// -------------------------------------------------------------------------------------------------

namespace pfc { namespace details {

template <typename ...T> struct validate_types {   // taken from stackoverflow.com/questions/12042824/how-to-write-a-type-trait-is-container-or-is-vector
};

template <typename T> using validate_container_types_t = validate_types <
   typename T::iterator,
   typename T::size_type,
   typename T::value_type,
   decltype (std::declval <T> ().begin ()),
   decltype (std::declval <T> ().end ()),
   decltype (std::declval <T> ().size ())
>;

} }   // namespace pfc::details

// -------------------------------------------------------------------------------------------------

namespace pfc { namespace details {

template <typename C> struct container_traits {
   using iterator   = typename C::iterator;
   using size_type  = typename C::size_type;
   using value_type = typename C::value_type;
};

template <typename T, int n> struct container_traits <T [n]> {
   using iterator   = T *;
   using size_type  = std::size_t;
   using value_type = T;
};

template <typename T> struct container_traits <T []> {
   using iterator   = T *;
   using size_type  = std::size_t;
   using value_type = T;
};

} }   // namespace pfc::details

// -------------------------------------------------------------------------------------------------

namespace pfc { namespace details {

template <typename T> struct is_integral : std::is_integral <T> {
};

template <> struct is_integral <bool> : std::false_type {
};

} }   // namespace pfc::details

// -------------------------------------------------------------------------------------------------

namespace pfc {

template <typename T> struct is_ratio final : std::false_type {
};

template <int num, int den> struct is_ratio <std::ratio <num, den>> final : std::true_type {
};

template <typename ratio_t> constexpr bool is_ratio_v {is_ratio <ratio_t>::value};

// -------------------------------------------------------------------------------------------------

template <typename T> struct type_identity final {
   using type = T;
};

template <typename T> using type_identity_t = typename type_identity <T>::type;

// -------------------------------------------------------------------------------------------------

template <typename T, typename U> using is_similar = std::is_same <std::decay_t <T>, std::decay_t <U>>;   // taken from stevedewhurst.com

template <typename T> using is_bool            = is_similar <T, bool>;
template <typename T> using is_integral        = details::is_integral <std::decay_t <T>>;
template <typename T> using not_is_string      = std::negation <is_similar <T, std::string>>;
template <typename T> using not_is_string_view = std::negation <is_similar <T, std::string_view>>;

template <typename T, typename U> constexpr bool is_similar_v {is_similar <T, U>::value};

template <typename T> constexpr bool is_bool_v              {is_bool <T>::value};
template <typename T> constexpr bool is_integral_v          {is_integral <T>::value};
template <typename T> constexpr bool is_integral_signed_v   {is_integral_v <T> && std::is_signed_v <T>};
template <typename T> constexpr bool is_integral_unsigned_v {is_integral_v <T> && std::is_unsigned_v <T>};

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

template <typename T> using floating_point = std::enable_if_t <std::is_floating_point_v <T>, T>;
template <typename T> using integral       = std::enable_if_t <pfc::is_integral_v       <T>, T>;

// -------------------------------------------------------------------------------------------------

template <typename T> struct limits_max final {
};

template <> struct limits_max <char>          final { constexpr static char          value {CHAR_MAX }; };
template <> struct limits_max <unsigned char> final { constexpr static unsigned char value {UCHAR_MAX}; };
template <> struct limits_max <int>           final { constexpr static int           value {INT_MAX  }; };
template <> struct limits_max <unsigned>      final { constexpr static unsigned      value {UINT_MAX }; };
template <> struct limits_max <float>         final { constexpr static float         value {FLT_MAX  }; };
template <> struct limits_max <double>        final { constexpr static double        value {DBL_MAX  }; };

template <typename T> constexpr static T /*auto*/ limits_max_v {limits_max <T>::value};   // !pwk: backward compatibility (e.g. for nvcc)

// -------------------------------------------------------------------------------------------------

namespace pfc {

template <typename T, typename = void> struct is_container : std::false_type {
};

template <typename T> struct is_container <T, std::void_t <details::validate_container_types_t <T>>> : std::true_type {
};

template <typename T> constexpr bool is_container_v {is_container <T>::value};

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

namespace pfc {

template <typename T, template <typename> typename ...R> constexpr bool satisfies_all_v  = (true  && ... &&  R <T>::value);   // taken from stevedewhurst.com
template <typename T, template <typename> typename ...R> constexpr bool satisfies_none_v = (true  && ... && !R <T>::value);   //
template <typename T, template <typename> typename ...R> constexpr bool satisfies_some_v = (false || ... ||  R <T>::value);   //

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

namespace pfc {

template <typename C> struct container_traits : details::container_traits <C> {
   using iterator   = typename details::container_traits <C>::iterator;
   using size_type  = typename details::container_traits <C>::size_type;
   using value_type = typename details::container_traits <C>::value_type;

   static constexpr value_type make_value_type () noexcept {
      return {};
   }
};

}   // namespace pfc

// -------------------------------------------------------------------------------------------------

//#undef  PFC_GENERATE_IS_XXX_ITERATOR
//#define PFC_GENERATE_IS_XXX_ITERATOR(cat)                                                                                               \
//   template <typename T> struct is_##cat##_iterator final                                                                               \
//      : public std::bool_constant <std::is_base_of_v <std::cat##_iterator_tag, typename std::iterator_traits <T>::iterator_category>> { \
//   };                                                                                                                                   \
//                                                                                                                                        \
//   template <typename T> constexpr bool /*auto*/ is_##cat##_iterator_v { /* !pwk: backward compatibility (e.g. for nvcc) */             \
//      pfc::is_##cat##_iterator <T>::value                                                                                               \
//   };
//
//PFC_GENERATE_IS_XXX_ITERATOR (input)
//PFC_GENERATE_IS_XXX_ITERATOR (output)
//PFC_GENERATE_IS_XXX_ITERATOR (forward)
//PFC_GENERATE_IS_XXX_ITERATOR (bidirectional)
//PFC_GENERATE_IS_XXX_ITERATOR (random_access)
//
//#undef PFC_GENERATE_IS_XXX_ITERATOR

// -------------------------------------------------------------------------------------------------

//#undef  PFC_GENERATE_HAS_XXX_ITERATOR
//#define PFC_GENERATE_HAS_XXX_ITERATOR(cat)                                                                                   \
//   template <typename T> struct has_##cat##_iterator final                                                                   \
//      : public std::bool_constant <pfc::is_##cat##_iterator_v <typename T::iterator>> {                                      \
//   };                                                                                                                        \
//                                                                                                                             \
//   template <typename T> constexpr bool /*auto*/ has_##cat##_iterator_v { /* !pwk: backward compatibility (e.g. for nvcc) */ \
//      pfc::has_##cat##_iterator <T>::value                                                                                   \
//   };
//
//PFC_GENERATE_HAS_XXX_ITERATOR (input)
//PFC_GENERATE_HAS_XXX_ITERATOR (output)
//PFC_GENERATE_HAS_XXX_ITERATOR (forward)
//PFC_GENERATE_HAS_XXX_ITERATOR (bidirectional)
//PFC_GENERATE_HAS_XXX_ITERATOR (random_access)
//
//#undef PFC_GENERATE_HAS_XXX_ITERATOR

// -------------------------------------------------------------------------------------------------

//}   // namespace pfc
