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

#include "util/utf8/unicodetext.h"

#include <string.h>                     // for memcpy, NULL, memcmp, etc
#include <algorithm>                    // for max

//#include "base/logging.h"               // for operator<<, CHECK, etc
//#include "base/stringprintf.h"          // for StringPrintf, StringAppendF
//#include "strings/stringpiece.h"        // for StringPiece, etc

#include "third_party/utf/utf.h"        // for isvalidcharntorune, etc
#include "util/utf8/unilib.h"    // for IsInterchangeValid, etc
#include "util/utf8/unilib_utf8_utils.h"    // for OneCharLen

static int CodepointDistance(const char* start, const char* end) {
  int n = 0;
  // Increment n on every non-trail-byte.
  for (const char* p = start; p < end; ++p) {
    n += (*reinterpret_cast<const signed char*>(p) >= -0x40);
  }
  return n;
}

static int CodepointCount(const char* utf8, int len) {
  return CodepointDistance(utf8, utf8 + len);
}

UnicodeText::const_iterator::difference_type
distance(const UnicodeText::const_iterator& first,
         const UnicodeText::const_iterator& last) {
  return CodepointDistance(first.it_, last.it_);
}

// ---------- Utility ----------

static int ConvertToInterchangeValid(char* start, int len) {
  // This routine is called only when we've discovered that a UTF-8 buffer
  // that was passed to CopyUTF8, TakeOwnershipOfUTF8, or PointToUTF8
  // was not interchange valid. This indicates a bug in the caller, and
  // a LOG(WARNING) is done in that case.
  // This is similar to CoerceToInterchangeValid, but it replaces each
  // structurally valid byte with a space, and each non-interchange
  // character with a space, even when that character requires more
  // than one byte in UTF8. E.g., "\xEF\xB7\x90" (U+FDD0) is
  // structurally valid UTF8, but U+FDD0 is not an interchange-valid
  // code point. The result should contain one space, not three.
  //
  // Since the conversion never needs to write more data than it
  // reads, it is safe to change the buffer in place. It returns the
  // number of bytes written.
  char* const in = start;
  char* out = start;
  char* const end = start + len;
  while (start < end) {
    int good = UniLib::SpanInterchangeValid(start, end - start);
    if (good > 0) {
      if (out != start) {
        memmove(out, start, good);
      }
      out += good;
      start += good;
      if (start == end) {
        break;
      }
    }
    // Is the current string invalid UTF8 or just non-interchange UTF8?
    char32 rune;
    int n;
    if (isvalidcharntorune(start, end - start, &rune, &n)) {
      // structurally valid UTF8, but not interchange valid
      start += n;  // Skip over the whole character.
    } else {  // bad UTF8
      start += 1;  // Skip over just one byte
    }
    *out++ = ' ';
  }
  return out - in;
}


// *************** Data representation **********

// Note: the copy constructor is undefined.

// After reserve(), resize(), or clear(), we're an owner, not an alias.

void UnicodeText::Repr::reserve(int new_capacity) {
  // If there's already enough capacity, and we're an owner, do nothing.
  if (capacity_ >= new_capacity && ours_) return;

  // Otherwise, allocate a new buffer.
  capacity_ = std::max(new_capacity, (3 * capacity_) / 2 + 20);
  char* new_data = new char[capacity_];

  // If there is an old buffer, copy it into the new buffer.
  if (data_) {
    memcpy(new_data, data_, size_);
    if (ours_) delete[] data_;  // If we owned the old buffer, free it.
  }
  data_ = new_data;
  ours_ = true;  // We own the new buffer.
  // size_ is unchanged.
}

void UnicodeText::Repr::resize(int new_size) {
  if (new_size == 0) {
    clear();
  } else {
    if (!ours_ || new_size > capacity_) reserve(new_size);
    // Clear the memory in the expanded part.
    if (size_ < new_size) memset(data_ + size_, 0, new_size - size_);
    size_ = new_size;
    ours_ = true;
  }
}

// This implementation of clear() deallocates the buffer if we're an owner.
// That's not strictly necessary; we could just set size_ to 0.
void UnicodeText::Repr::clear() {
  if (ours_) delete[] data_;
  data_ = nullptr;
  size_ = capacity_ = 0;
  ours_ = true;
}

void UnicodeText::Repr::Copy(const char* data, int size) {
  resize(size);
  memcpy(data_, data, size);
}

void UnicodeText::Repr::TakeOwnershipOf(char* data, int size, int capacity) {
  if (data == data_) return;  // We already own this memory. (Weird case.)
  if (ours_ && data_) delete[] data_;  // If we owned the old buffer, free it.
  data_ = data;
  size_ = size;
  capacity_ = capacity;
  ours_ = true;
}

void UnicodeText::Repr::PointTo(const char* data, int size) {
  if (ours_ && data_) delete[] data_;  // If we owned the old buffer, free it.
  data_ = const_cast<char*>(data);
  size_ = size;
  capacity_ = size;
  ours_ = false;
}

void UnicodeText::Repr::append(const char* bytes, int byte_length) {
  reserve(size_ + byte_length);
  memcpy(data_ + size_, bytes, byte_length);
  size_ += byte_length;
}

string UnicodeText::Repr::DebugString() const {
  return tensorflow::strings::Printf("{Repr %p data=%p size=%d capacity=%d %s}",
                      this,
                      data_, size_, capacity_,
                      ours_ ? "Owned" : "Alias");
}



// *************** UnicodeText ******************

// ----- Constructors -----

// Default constructor
UnicodeText::UnicodeText() {
}

// Copy constructor
UnicodeText::UnicodeText(const UnicodeText& src) {
  Copy(src);
}

// Substring constructor
UnicodeText::UnicodeText(const UnicodeText::const_iterator& first,
                         const UnicodeText::const_iterator& last) {
  CHECK(first <= last) << " Incompatible iterators";
  repr_.append(first.it_, last.it_ - first.it_);
}

string UnicodeText::UTF8Substring(const const_iterator& first,
                                  const const_iterator& last) {
  CHECK(first <= last) << " Incompatible iterators";
  return string(first.it_, last.it_ - first.it_);
}


// ----- Copy -----

UnicodeText& UnicodeText::operator=(const UnicodeText& src) {
  if (this != &src) {
    Copy(src);
  }
  return *this;
}

UnicodeText& UnicodeText::Copy(const UnicodeText& src) {
  repr_.Copy(src.repr_.data_, src.repr_.size_);
  return *this;
}

UnicodeText& UnicodeText::CopyUTF8(const char* buffer, int byte_length) {
  repr_.Copy(buffer, byte_length);
  if (!UniLib:: IsInterchangeValid(buffer, byte_length)) {
    LOG(WARNING) << "UTF-8 buffer is not interchange-valid.";
    repr_.size_ = ConvertToInterchangeValid(repr_.data_, byte_length);
  }
  return *this;
}

UnicodeText& UnicodeText::UnsafeCopyUTF8(const char* buffer,
                                           int byte_length) {
  repr_.Copy(buffer, byte_length);
  return *this;
}

// ----- TakeOwnershipOf  -----

UnicodeText& UnicodeText::TakeOwnershipOfUTF8(char* buffer,
                                              int byte_length,
                                              int byte_capacity) {
  repr_.TakeOwnershipOf(buffer, byte_length, byte_capacity);
  if (!UniLib:: IsInterchangeValid(buffer, byte_length)) {
    LOG(WARNING) << "UTF-8 buffer is not interchange-valid.";
    repr_.size_ = ConvertToInterchangeValid(repr_.data_, byte_length);
  }
  return *this;
}

UnicodeText& UnicodeText::UnsafeTakeOwnershipOfUTF8(char* buffer,
                                                    int byte_length,
                                                    int byte_capacity) {
  repr_.TakeOwnershipOf(buffer, byte_length, byte_capacity);
  return *this;
}

// ----- PointTo -----

UnicodeText& UnicodeText::PointToUTF8(const char* buffer, int byte_length) {
  if (UniLib:: IsInterchangeValid(buffer, byte_length)) {
    repr_.PointTo(buffer, byte_length);
  } else {
    LOG(WARNING) << "UTF-8 buffer is not interchange-valid.";
    repr_.Copy(buffer, byte_length);
    repr_.size_ = ConvertToInterchangeValid(repr_.data_, byte_length);
  }
  return *this;
}

UnicodeText& UnicodeText::UnsafePointToUTF8(const char* buffer,
                                          int byte_length) {
  repr_.PointTo(buffer, byte_length);
  return *this;
}

UnicodeText& UnicodeText::PointTo(const UnicodeText& src) {
  repr_.PointTo(src.repr_.data_, src.repr_.size_);
  return *this;
}

UnicodeText& UnicodeText::PointTo(const const_iterator &first,
                                  const const_iterator &last) {
  CHECK(first <= last) << " Incompatible iterators";
  repr_.PointTo(first.utf8_data(), last.utf8_data() - first.utf8_data());
  return *this;
}

// ----- Append -----

UnicodeText& UnicodeText::append(const UnicodeText& u) {
  repr_.append(u.repr_.data_, u.repr_.size_);
  return *this;
}

UnicodeText& UnicodeText::append(const const_iterator& first,
                                 const const_iterator& last) {
  CHECK(first <= last) << " Incompatible iterators";
  repr_.append(first.it_, last.it_ - first.it_);
  return *this;
}

UnicodeText& UnicodeText::UnsafeAppendUTF8(const char* utf8, int len) {
  repr_.append(utf8, len);
  return *this;
}

// ----- substring searching -----

UnicodeText::const_iterator UnicodeText::find(const UnicodeText& look,
                                              const_iterator start_pos) const {
  CHECK_GE(start_pos.utf8_data(), utf8_data());
  CHECK_LE(start_pos.utf8_data(), utf8_data() + utf8_length());
  return UnsafeFind(look, start_pos);
}

UnicodeText::const_iterator UnicodeText::find(const UnicodeText& look) const {
  return UnsafeFind(look, begin());
}

UnicodeText::const_iterator UnicodeText::UnsafeFind(
    const UnicodeText& look, const_iterator start_pos) const {
  // Due to the magic of the UTF8 encoding, searching for a sequence of
  // letters is equivalent to substring search.
  StringPiece searching(utf8_data(), utf8_length());
  StringPiece look_piece(look.utf8_data(), look.utf8_length());
  LOG(FATAL) << "Not implemented";
  //StringPiece::size_type found =
  //    searching.find(look_piece, start_pos.utf8_data() - utf8_data());
  StringPiece::size_type found = StringPiece::npos;
  if (found == StringPiece::npos) return end();
  return const_iterator(utf8_data() + found);
}

bool UnicodeText::HasReplacementChar() const {
  // Equivalent to:
  //   UnicodeText replacement_char;
  //   replacement_char.push_back(0xFFFD);
  //   return find(replacement_char) != end();
  StringPiece searching(utf8_data(), utf8_length());
  StringPiece looking_for("\xEF\xBF\xBD", 3);
  LOG(FATAL) << "Not implemented";
  //return searching.find(looking_for) != StringPiece::npos;
  return false;
}

// ----- other methods -----

// Clear operator
void UnicodeText::clear() {
  repr_.clear();
}

// Destructor
UnicodeText::~UnicodeText() {}


void UnicodeText::push_back(char32 c) {
  if (UniLib::IsValidCodepoint(c)) {
    char buf[UTFmax];
    int len = runetochar(buf, &c);
    if (UniLib::IsInterchangeValid(buf, len)) {
      repr_.append(buf, len);
    } else {
      LOG(WARNING) << "Unicode value 0x" << std::hex << c
                  << " is not valid for interchange";
      repr_.append(" ", 1);
    }
  } else {
    LOG(WARNING) << "Illegal Unicode value: 0x" << std::hex << c;
    repr_.append(" ", 1);
  }
}

int UnicodeText::size() const {
  return CodepointCount(repr_.data_, repr_.size_);
}

bool operator==(const UnicodeText& lhs, const UnicodeText& rhs) {
  if (&lhs == &rhs) return true;
  if (lhs.repr_.size_ != rhs.repr_.size_) return false;
  return memcmp(lhs.repr_.data_, rhs.repr_.data_, lhs.repr_.size_) == 0;
}

string UnicodeText::DebugString() const {
  return tensorflow::strings::Printf("{UnicodeText %p chars=%d repr=%s}",
                      this,
                      size(),
                      repr_.DebugString().c_str());
}


// ******************* UnicodeText::const_iterator *********************

// The implementation of const_iterator would be nicer if it
// inherited from boost::iterator_facade
// (http://boost.org/libs/iterator/doc/iterator_facade.html).

UnicodeText::const_iterator::const_iterator() : it_(nullptr) {}

UnicodeText::const_iterator::const_iterator(const const_iterator& other)
    : it_(other.it_) {
}

UnicodeText::const_iterator&
UnicodeText::const_iterator::operator=(const const_iterator& other) {
  if (&other != this)
    it_ = other.it_;
  return *this;
}

UnicodeText::const_iterator UnicodeText::begin() const {
  return const_iterator(repr_.data_);
}

UnicodeText::const_iterator UnicodeText::end() const {
  return const_iterator(repr_.data_ + repr_.size_);
}

bool operator<(const UnicodeText::const_iterator& lhs,
               const UnicodeText::const_iterator& rhs) {
  return lhs.it_ < rhs.it_;
}

char32 UnicodeText::const_iterator::operator*() const {
  // (We could call chartorune here, but that does some
  // error-checking, and we're guaranteed that our data is valid
  // UTF-8. Also, we expect this routine to be called very often. So
  // for speed, we do the calculation ourselves.)

  // Convert from UTF-8
  int byte1 = it_[0];
  if (byte1 < 0x80)
    return byte1;

  int byte2 = it_[1];
  if (byte1 < 0xE0)
    return ((byte1 & 0x1F) << 6)
          | (byte2 & 0x3F);

  int byte3 = it_[2];
  if (byte1 < 0xF0)
    return ((byte1 & 0x0F) << 12)
         | ((byte2 & 0x3F) << 6)
         |  (byte3 & 0x3F);

  int byte4 = it_[3];
  return ((byte1 & 0x07) << 18)
       | ((byte2 & 0x3F) << 12)
       | ((byte3 & 0x3F) << 6)
       |  (byte4 & 0x3F);
}

UnicodeText::const_iterator& UnicodeText::const_iterator::operator++() {
  it_ += UniLib::OneCharLen(it_);
  return *this;
}

UnicodeText::const_iterator& UnicodeText::const_iterator::operator--() {
  while (UniLib::IsTrailByte(*--it_));
  return *this;
}

int UnicodeText::const_iterator::get_utf8(char* utf8_output) const {
  utf8_output[0] = it_[0]; if (it_[0] < 0x80) return 1;
  utf8_output[1] = it_[1]; if (it_[0] < 0xE0) return 2;
  utf8_output[2] = it_[2]; if (it_[0] < 0xF0) return 3;
  utf8_output[3] = it_[3];
  return 4;
}

string UnicodeText::const_iterator::get_utf8_string() const {
  return string(utf8_data(), utf8_length());
}

int UnicodeText::const_iterator::utf8_length() const {
  if (it_[0] < 0x80) {
    return 1;
  } else if (it_[0] < 0xE0) {
    return 2;
  } else if (it_[0] < 0xF0) {
    return 3;
  } else {
    return 4;
  }
}

UnicodeText::const_iterator UnicodeText::MakeIterator(const char* p) const {
  CHECK(p != nullptr);
  const char* start = utf8_data();
  int len = utf8_length();
  const char* end = start + len;
  CHECK(p >= start);
  CHECK(p <= end);
  CHECK(p == end || !UniLib::IsTrailByte(*p));
  return const_iterator(p);
}

string UnicodeText::const_iterator::DebugString() const {
  return tensorflow::strings::Printf("{iter %p}", it_);
}


// *************************** Utilities *************************

string CodepointString(const UnicodeText& t) {
  string s;
  UnicodeText::const_iterator it = t.begin(), end = t.end();
  while (it != end) tensorflow::strings::Appendf(&s, "%X ", *it++);
  return s;
}
