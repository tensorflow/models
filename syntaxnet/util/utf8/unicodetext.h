/**
 * Copyright 2010 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef UTIL_UTF8_PUBLIC_UNICODETEXT_H_
#define UTIL_UTF8_PUBLIC_UNICODETEXT_H_

#include <stddef.h>                     // for NULL, ptrdiff_t
#include <iterator>                     // for bidirectional_iterator_tag, etc
#include <string>                       // for string
#include <utility>                      // for pair

#include "syntaxnet/base.h"

// ***************************** UnicodeText **************************
//
// A UnicodeText object is a container for a sequence of Unicode
// codepoint values. It has default, copy, and assignment constructors.
// Data can be appended to it from another UnicodeText, from
// iterators, or from a single codepoint.
//
// The internal representation of the text is UTF-8. Since UTF-8 is a
// variable-width format, UnicodeText does not provide random access
// to the text, and changes to the text are permitted only at the end.
//
// The UnicodeText class defines a const_iterator. The dereferencing
// operator (*) returns a codepoint (char32). The iterator is a
// bidirectional, read-only iterator. It becomes invalid if the text
// is changed.
//
// There are methods for appending and retrieving UTF-8 data directly.
// The 'utf8_data' method returns a const char* that contains the
// UTF-8-encoded version of the text; 'utf8_length' returns the number
// of bytes in the UTF-8 data. An iterator's 'get' method stores up to
// 4 bytes of UTF-8 data in a char array and returns the number of
// bytes that it stored.
//
// Codepoints are integers in the range [0, 0xD7FF] or [0xE000,
// 0x10FFFF], but UnicodeText has the additional restriction that it
// can contain only those characters that are valid for interchange on
// the Web. This excludes all of the control codes except for carriage
// return, line feed, and horizontal tab.  It also excludes
// non-characters, but codepoints that are in the Private Use regions
// are allowed, as are codepoints that are unassigned. (See the
// Unicode reference for details.) The function UniLib::IsInterchangeValid
// can be used as a test for this property.
//
// UnicodeTexts are safe. Every method that constructs or modifies a
// UnicodeText tests for interchange-validity, and will substitute a
// space for the invalid data. Such cases are reported via
// LOG(WARNING).
//
// MEMORY MANAGEMENT: copy, take ownership, or point to
//
// A UnicodeText is either an "owner", meaning that it owns the memory
// for the data buffer and will free it when the UnicodeText is
// destroyed, or it is an "alias", meaning that it does not.
//
// There are three methods for storing UTF-8 data in a UnicodeText:
//
// CopyUTF8(buffer, len) copies buffer.
//
// TakeOwnershipOfUTF8(buffer, size, capacity) takes ownership of buffer.
//
// PointToUTF8(buffer, size) creates an alias pointing to buffer.
//
// All three methods perform a validity check on the buffer. There are
// private, "unsafe" versions of these functions that bypass the
// validity check. They are used internally and by friend-functions
// that are handling UTF-8 data that has already been validated.
//
// The purpose of an alias is to avoid making an unnecessary copy of a
// UTF-8 buffer while still providing access to the Unicode values
// within that text through iterators or the fast scanners that are
// based on UTF-8 state tables. The lifetime of an alias must not
// exceed the lifetime of the buffer from which it was constructed.
//
// The semantics of an alias might be described as "copy on write or
// repair." The source data is never modified. If push_back() or
// append() is called on an alias, a copy of the data will be created,
// and the UnicodeText will become an owner. If clear() is called on
// an alias, it becomes an (empty) owner.
//
// The copy constructor and the assignment operator produce an owner.
// That is, after direct initialization ("UnicodeText x(y);") or copy
// initialization ("UnicodeText x = y;") x will be an owner, even if y
// was an alias. The assignment operator ("x = y;") also produces an
// owner unless x and y are the same object and y is an alias.
//
// Aliases should be used with care. If the source from which an alias
// was created is freed, or if the contents are changed, while the
// alias is still in use, fatal errors could result. But it can be
// quite useful to have a UnicodeText "window" through which to see a
// UTF-8 buffer without having to pay the price of making a copy.
//
// UTILITIES
//
// The interfaces in util/utf8/public/textutils.h provide higher-level
// utilities for dealing with UnicodeTexts, including routines for
// creating UnicodeTexts (both owners and aliases) from UTF-8 buffers or
// strings, creating strings from UnicodeTexts, normalizing text for
// efficient matching or display, and others.

class UnicodeText {
 public:
  class const_iterator;

  typedef char32 value_type;

  // Constructors. These always produce owners.
  UnicodeText();  // Create an empty text.
  UnicodeText(const UnicodeText& src);  // copy constructor
  // Construct a substring (copies the data).
  UnicodeText(const const_iterator& first, const const_iterator& last);

  // Assignment operator. This copies the data and produces an owner
  // unless this == &src, e.g., "x = x;", which is a no-op.
  UnicodeText& operator=(const UnicodeText& src);

  // x.Copy(y) copies the data from y into x.
  UnicodeText& Copy(const UnicodeText& src);
  inline UnicodeText& assign(const UnicodeText& src) { return Copy(src); }

  // x.PointTo(y) changes x so that it points to y's data.
  // It does not copy y or take ownership of y's data.
  UnicodeText& PointTo(const UnicodeText& src);
  UnicodeText& PointTo(const const_iterator& first,
                       const const_iterator& last);

  ~UnicodeText();

  void clear();  // Clear text.
  bool empty() const { return repr_.size_ == 0; }  // Test if text is empty.

  // Add a codepoint to the end of the text.
  // If the codepoint is not interchange-valid, add a space instead
  // and log a warning.
  void push_back(char32 codepoint);

  // Generic appending operation.
  // iterator_traits<ForwardIterator>::value_type must be implicitly
  // convertible to char32. Typical uses of this method might include:
  //     char32 chars[] = {0x1, 0x2, ...};
  //     vector<char32> more_chars = ...;
  //     utext.append(chars, chars+arraysize(chars));
  //     utext.append(more_chars.begin(), more_chars.end());
  template<typename ForwardIterator>
  UnicodeText& append(ForwardIterator first, const ForwardIterator last) {
    while (first != last) { push_back(*first++); }
    return *this;
  }

  // A specialization of the generic append() method.
  UnicodeText& append(const const_iterator& first, const const_iterator& last);

  // An optimization of append(source.begin(), source.end()).
  UnicodeText& append(const UnicodeText& source);

  int size() const;  // the number of Unicode characters (codepoints)

  friend bool operator==(const UnicodeText& lhs, const UnicodeText& rhs);
  friend bool operator!=(const UnicodeText& lhs, const UnicodeText& rhs);

  class const_iterator {
    typedef const_iterator CI;
   public:
    typedef std::bidirectional_iterator_tag iterator_category;
    typedef char32 value_type;
    typedef ptrdiff_t difference_type;
    typedef void pointer;  // (Not needed.)
    typedef const char32 reference;  // (Needed for const_reverse_iterator)

    // Iterators are default-constructible.
    const_iterator();

    // It's safe to make multiple passes over a UnicodeText.
    const_iterator(const const_iterator& other);
    const_iterator& operator=(const const_iterator& other);

    char32 operator*() const;  // Dereference

    const_iterator& operator++();  // Advance (++iter)
    const_iterator operator++(int) {  // (iter++)
      const_iterator result(*this);
      ++*this;
      return result;
    }

    const_iterator& operator--();  // Retreat (--iter)
    const_iterator operator--(int) {  // (iter--)
      const_iterator result(*this);
      --*this;
      return result;
    }

    // We love relational operators.
    friend bool operator==(const CI& lhs, const CI& rhs) {
      return lhs.it_ == rhs.it_; }
    friend bool operator!=(const CI& lhs, const CI& rhs) {
      return !(lhs == rhs); }
    friend bool operator<(const CI& lhs, const CI& rhs);
    friend bool operator>(const CI& lhs, const CI& rhs) {
      return rhs < lhs; }
    friend bool operator<=(const CI& lhs, const CI& rhs) {
      return !(rhs < lhs); }
    friend bool operator>=(const CI& lhs, const CI& rhs) {
      return !(lhs < rhs); }

    friend difference_type distance(const CI& first, const CI& last);

    // UTF-8-specific methods
    // Store the UTF-8 encoding of the current codepoint into buf,
    // which must be at least 4 bytes long. Return the number of
    // bytes written.
    int get_utf8(char* buf) const;
    // Return the UTF-8 character that the iterator points to.
    string get_utf8_string() const;
    // Return the byte length of the UTF-8 character the iterator points to.
    int utf8_length() const;
    // Return the iterator's pointer into the UTF-8 data.
    const char* utf8_data() const { return it_; }

    string DebugString() const;

   private:
    friend class UnicodeText;
    friend class UnicodeTextUtils;
    friend class UTF8StateTableProperty;
    explicit const_iterator(const char* it) : it_(it) {}

    const char* it_;
  };

  const_iterator begin() const;
  const_iterator end() const;

  class const_reverse_iterator : public std::reverse_iterator<const_iterator> {
   public:
    explicit const_reverse_iterator(const_iterator it) :
        std::reverse_iterator<const_iterator>(it) {}
    const char* utf8_data() const {
      const_iterator tmp_it = base();
      return (--tmp_it).utf8_data();
    }
    int get_utf8(char* buf) const {
      const_iterator tmp_it = base();
      return (--tmp_it).get_utf8(buf);
    }
    string get_utf8_string() const {
      const_iterator tmp_it = base();
      return (--tmp_it).get_utf8_string();
    }
    int utf8_length() const {
      const_iterator tmp_it = base();
      return (--tmp_it).utf8_length();
    }
  };
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }

  // Substring searching.  Returns the beginning of the first
  // occurrence of "look", or end() if not found.
  const_iterator find(const UnicodeText& look, const_iterator start_pos) const;
  // Equivalent to find(look, begin())
  const_iterator find(const UnicodeText& look) const;

  // Returns whether this contains the character U+FFFD.  This can
  // occur, for example, if the input to Encodings::Decode() had byte
  // sequences that were invalid in the source encoding.
  bool HasReplacementChar() const;

  // UTF-8-specific methods
  //
  // Return the data, length, and capacity of UTF-8-encoded version of
  // the text. Length and capacity are measured in bytes.
  const char* utf8_data() const { return repr_.data_; }
  int utf8_length() const { return repr_.size_; }
  int utf8_capacity() const { return repr_.capacity_; }

  // Return the UTF-8 data as a string.
  static string UTF8Substring(const const_iterator& first,
                              const const_iterator& last);

  // There are three methods for initializing a UnicodeText from UTF-8
  // data. They vary in details of memory management. In all cases,
  // the data is tested for interchange-validity. If it is not
  // interchange-valid, a LOG(WARNING) is issued, and each
  // structurally invalid byte and each interchange-invalid codepoint
  // is replaced with a space.

  // x.CopyUTF8(buf, len) copies buf into x.
  UnicodeText& CopyUTF8(const char* utf8_buffer, int byte_length);

  // x.TakeOwnershipOfUTF8(buf, len, capacity). x takes ownership of
  // buf. buf is not copied.
  UnicodeText& TakeOwnershipOfUTF8(char* utf8_buffer,
                                   int byte_length,
                                   int byte_capacity);

  // x.PointToUTF8(buf,len) changes x so that it points to buf
  // ("becomes an alias"). It does not take ownership or copy buf.
  // If the buffer is not valid, this has the same effect as
  // CopyUTF8(utf8_buffer, byte_length).
  UnicodeText& PointToUTF8(const char* utf8_buffer, int byte_length);

  // Occasionally it is necessary to use functions that operate on the
  // pointer returned by utf8_data(). MakeIterator(p) provides a way
  // to get back to the UnicodeText level. It uses CHECK to ensure
  // that p is a pointer within this object's UTF-8 data, and that it
  // points to the beginning of a character.
  const_iterator MakeIterator(const char* p) const;

  string DebugString() const;

 private:
  friend class const_iterator;
  friend class UnicodeTextUtils;

  class Repr {  // A byte-string.
   public:
    char* data_;
    int size_;
    int capacity_;
    bool ours_;  // Do we own data_?

    Repr() : data_(nullptr), size_(0), capacity_(0), ours_(true) {}
    ~Repr() { if (ours_) delete[] data_; }

    void clear();
    void reserve(int capacity);
    void resize(int size);

    void append(const char* bytes, int byte_length);
    void Copy(const char* data, int size);
    void TakeOwnershipOf(char* data, int size, int capacity);
    void PointTo(const char* data, int size);

    string DebugString() const;

   private:
    Repr& operator=(const Repr&);
    Repr(const Repr& other);
  };

  Repr repr_;

  // UTF-8-specific private methods.
  // These routines do not perform a validity check when compiled
  // in opt mode.
  // It is an error to call these methods with UTF-8 data that
  // is not interchange-valid.
  //
  UnicodeText& UnsafeCopyUTF8(const char* utf8_buffer, int byte_length);
  UnicodeText& UnsafeTakeOwnershipOfUTF8(
      char* utf8_buffer, int byte_length, int byte_capacity);
  UnicodeText& UnsafePointToUTF8(const char* utf8_buffer, int byte_length);
  UnicodeText& UnsafeAppendUTF8(const char* utf8_buffer, int byte_length);
  const_iterator UnsafeFind(const UnicodeText& look,
                            const_iterator start_pos) const;
};

bool operator==(const UnicodeText& lhs, const UnicodeText& rhs);

inline bool operator!=(const UnicodeText& lhs, const UnicodeText& rhs) {
  return !(lhs == rhs);
}

// UnicodeTextRange is a pair of iterators, useful for specifying text
// segments. If the iterators are ==, the segment is empty.
typedef pair<UnicodeText::const_iterator,
             UnicodeText::const_iterator> UnicodeTextRange;

inline bool UnicodeTextRangeIsEmpty(const UnicodeTextRange& r) {
  return r.first == r.second;
}


// *************************** Utilities *************************

// A factory function for creating a UnicodeText from a buffer of
// UTF-8 data. The new UnicodeText takes ownership of the buffer. (It
// is an "owner.")
//
// Each byte that is structurally invalid will be replaced with a
// space. Each codepoint that is interchange-invalid will also be
// replaced with a space, even if the codepoint was represented with a
// multibyte sequence in the UTF-8 data.
//
inline UnicodeText MakeUnicodeTextAcceptingOwnership(
    char* utf8_buffer, int byte_length, int byte_capacity) {
  return UnicodeText().TakeOwnershipOfUTF8(
      utf8_buffer, byte_length, byte_capacity);
}

// A factory function for creating a UnicodeText from a buffer of
// UTF-8 data. The new UnicodeText does not take ownership of the
// buffer. (It is an "alias.")
//
inline UnicodeText MakeUnicodeTextWithoutAcceptingOwnership(
    const char* utf8_buffer, int byte_length) {
  return UnicodeText().PointToUTF8(utf8_buffer, byte_length);
}

// Create a UnicodeText from a UTF-8 string or buffer.
//
// If do_copy is true, then a copy of the string is made. The copy is
// owned by the resulting UnicodeText object and will be freed when
// the object is destroyed. This UnicodeText object is referred to
// as an "owner."
//
// If do_copy is false, then no copy is made. The resulting
// UnicodeText object does NOT take ownership of the string; in this
// case, the lifetime of the UnicodeText object must not exceed the
// lifetime of the string. This Unicodetext object is referred to as
// an "alias." This is the same as MakeUnicodeTextWithoutAcceptingOwnership.
//
// If the input string does not contain valid UTF-8, then a copy is
// made (as if do_copy were true) and coerced to valid UTF-8 by
// replacing each invalid byte with a space.
//
inline UnicodeText UTF8ToUnicodeText(const char* utf8_buf, int len,
                                     bool do_copy) {
  UnicodeText t;
  if (do_copy) {
    t.CopyUTF8(utf8_buf, len);
  } else {
    t.PointToUTF8(utf8_buf, len);
  }
  return t;
}

inline UnicodeText UTF8ToUnicodeText(const string& utf_string, bool do_copy) {
  return UTF8ToUnicodeText(utf_string.data(), utf_string.size(), do_copy);
}

inline UnicodeText UTF8ToUnicodeText(const char* utf8_buf, int len) {
  return UTF8ToUnicodeText(utf8_buf, len, true);
}
inline UnicodeText UTF8ToUnicodeText(const string& utf8_string) {
  return UTF8ToUnicodeText(utf8_string, true);
}

// Return a string containing the UTF-8 encoded version of all the
// Unicode characters in t.
inline string UnicodeTextToUTF8(const UnicodeText& t) {
  return string(t.utf8_data(), t.utf8_length());
}


// For debugging.  Return a string of integers, written in uppercase
// hex (%X), corresponding to the codepoints within the text. Each
// integer is followed by a space. E.g., "61 62 6A 3005 ".
string CodepointString(const UnicodeText& t);

#endif  // UTIL_UTF8_PUBLIC_UNICODETEXT_H_
