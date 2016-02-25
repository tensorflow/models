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

#include <iterator>
#include <set>

#include "gtest/gtest.h"
#include "third_party/utf/utf.h"
#include "util/utf8/unilib.h"

namespace {

template <typename T, size_t N>
char (&ArraySizeHelper(T (&array)[N]))[N];
#define arraysize(array) (sizeof(ArraySizeHelper(array)))

class UnicodeTextTest : public testing::Test {
 protected:
  UnicodeTextTest() : empty_text_() {
    const char32 text[] = {0x1C0, 0x4E8C, 0xD7DB, 0x34, 0x1D11E};
    // Construct a UnicodeText from those codepoints.
    text_.append(&text[0], text + arraysize(text));
  }

  UnicodeText empty_text_;
  UnicodeText text_;
};

TEST(UnicodeTextTest, Ownership) {
  const string src =  "\u304A\u00B0\u106B";
  {
    string s = src;
    char* sbuf = new char[s.size()];
    memcpy(sbuf, s.data(), s.size());
    UnicodeText owned;
    owned.TakeOwnershipOfUTF8(sbuf, s.size(), s.size());
    EXPECT_EQ(owned.utf8_data(), sbuf);
    s.clear();
    // owned should be OK even after s has been cleared.
    UnicodeText::const_iterator it = owned.begin();
    EXPECT_EQ(*it++, 0x304A);
    EXPECT_EQ(*it++, 0x00B0);
    EXPECT_EQ(*it++, 0x106B);
    CHECK(it == owned.end());
  }

  {
    UnicodeText owner;
    {  // Create a new scope for s.
      string s = src;
      char* sbuf = new char[s.size()];
      memcpy(sbuf, s.data(), s.size());
      UnicodeText t;
      t.TakeOwnershipOfUTF8(sbuf, s.size(), s.size());
      EXPECT_EQ(t.utf8_data(), sbuf);
      owner = t;  // Copies the data
      EXPECT_NE(owner.utf8_data(), sbuf);
    }
    // owner should be OK even after s has gone out of scope
    UnicodeText::const_iterator it = owner.begin();
    EXPECT_EQ(*it++, 0x304A);
    EXPECT_EQ(*it++, 0x00B0);
    EXPECT_EQ(*it++, 0x106B);
    CHECK(it == owner.end());
  }

  {
    UnicodeText alias;
    alias.PointToUTF8(src.data(), src.size());
    EXPECT_EQ(alias.utf8_data(), src.data());
    UnicodeText::const_iterator it = alias.begin();
    EXPECT_EQ(*it++, 0x304A);
    EXPECT_EQ(*it++, 0x00B0);
    EXPECT_EQ(*it++, 0x106B);
    CHECK(it == alias.end());

    UnicodeText t = alias;  // Copy initialization copies the data.
    EXPECT_NE(t.utf8_data(), alias.utf8_data());

    UnicodeText t2;
    t2 = alias;  // Assignment copies the data.
    EXPECT_NE(t2.utf8_data(), alias.utf8_data());

    // Preserve an alias.
    t.PointTo(alias); // This does not copy the data.
    EXPECT_EQ(t.utf8_data(), alias.utf8_data());

    t.push_back(0x0020); // Modify the alias
    EXPECT_NE(t.utf8_data(), alias.utf8_data()); // It's no longer an alias.
  }
}

class IteratorTest : public UnicodeTextTest {};

TEST_F(IteratorTest, Iterates) {
  UnicodeText::const_iterator iter = text_.begin();
  EXPECT_EQ(0x1C0, *iter);
  EXPECT_EQ(&iter, &++iter);  // operator++ returns *this.
  EXPECT_EQ(0x4E8C, *iter++);
  EXPECT_EQ(0xD7DB, *iter);
  // Make sure you can dereference more than once.
  EXPECT_EQ(0xD7DB, *iter);
  EXPECT_EQ(0x34, *++iter);
  EXPECT_EQ(0x1D11E, *++iter);
  ASSERT_TRUE(iter != text_.end());
  iter++;
  EXPECT_TRUE(iter == text_.end());
}

TEST_F(IteratorTest, Reverse) {
  UnicodeText::const_reverse_iterator iter = text_.rbegin();
  EXPECT_EQ(0x1D11E, *iter);
  EXPECT_EQ(&iter, &++iter);  // operator++ returns *this.
  EXPECT_EQ(0x34, *iter++);
  EXPECT_EQ(0xD7DB, *iter);
  // Make sure you can dereference more than once.
  EXPECT_EQ(0xD7DB, *iter);
  EXPECT_EQ(0x4E8C, *++iter);
  EXPECT_EQ(0x1C0, *++iter);
  ASSERT_TRUE(iter != text_.rend());
  iter++;
  EXPECT_TRUE(iter == text_.rend());
}

TEST_F(IteratorTest, MultiPass) {
  // Also tests Default Constructible and Assignable.
  UnicodeText::const_iterator i1, i2;
  i1 = text_.begin();
  i2 = i1;
  EXPECT_EQ(0x4E8C, *++i1);
  EXPECT_TRUE(i1 != i2);
  EXPECT_EQ(0x1C0, *i2);
  ++i2;
  EXPECT_TRUE(i1 == i2);
  EXPECT_EQ(0x4E8C, *i2);
}

TEST_F(IteratorTest, ReverseIterates) {
  UnicodeText::const_iterator iter = text_.end();
  EXPECT_TRUE(iter == text_.end());
  iter--;
  ASSERT_TRUE(iter != text_.end());
  EXPECT_EQ(0x1D11E, *iter--);
  EXPECT_EQ(0x34, *iter);
  EXPECT_EQ(0xD7DB, *--iter);
  // Make sure you can dereference more than once.
  EXPECT_EQ(0xD7DB, *iter);
  --iter;
  EXPECT_EQ(0x4E8C, *iter--);
  EXPECT_EQ(0x1C0, *iter);
  EXPECT_TRUE(iter == text_.begin());
}

TEST_F(IteratorTest, Comparable) {
  UnicodeText::const_iterator i1, i2;
  i1 = text_.begin();
  i2 = i1;
  ++i2;

  EXPECT_TRUE(i1 < i2);
  EXPECT_TRUE(text_.begin() <= i1);
  EXPECT_FALSE(i1 >= i2);
  EXPECT_FALSE(i1 > text_.end());
}

TEST_F(IteratorTest, Advance) {
  UnicodeText::const_iterator iter = text_.begin();
  EXPECT_EQ(0x1C0, *iter);
  std::advance(iter, 4);
  EXPECT_EQ(0x1D11E, *iter);
  ++iter;
  EXPECT_TRUE(iter == text_.end());
}

TEST_F(IteratorTest, Distance) {
  UnicodeText::const_iterator iter = text_.begin();
  EXPECT_EQ(0, distance(text_.begin(), iter));
  EXPECT_EQ(5, distance(iter, text_.end()));
  ++iter;
  ++iter;
  EXPECT_EQ(2, distance(text_.begin(), iter));
  EXPECT_EQ(3, distance(iter, text_.end()));
  ++iter;
  ++iter;
  EXPECT_EQ(4, distance(text_.begin(), iter));
  ++iter;
  EXPECT_EQ(0, distance(iter, text_.end()));
}

TEST_F(IteratorTest, Encode) {
  const string utf8 = "\xC7\x80"
                      "\xE4\xBA\x8C"
                      "\xED\x9F\x9B"
                      "\x34"
                      "\xF0\x9D\x84\x9E";
  const int lengths[] = {2, 3, 3, 1, 4};
  EXPECT_EQ(text_.size(), 5);
  EXPECT_EQ(text_.utf8_length(), 13);
  EXPECT_TRUE(memcmp(text_.utf8_data(), utf8.data(), text_.utf8_length())
              == 0);

  {
    // Test the iterator
    UnicodeText::const_iterator iter = text_.begin(), end = text_.end();
    const char* u = utf8.data();
    int i = 0;
    while (iter != end) {
      char buf[5];
      int n = iter.get_utf8(buf);
      buf[n] = '\0';
      EXPECT_TRUE(strncmp(buf, u, n) == 0);
      EXPECT_EQ(buf, iter.get_utf8_string());
      EXPECT_EQ(lengths[i], iter.utf8_length());
      u += n;
      iter++;
      i++;
    }
  }

  {
    // Test the reverse_iterator
    UnicodeText::const_reverse_iterator iter = text_.rbegin();
    UnicodeText::const_reverse_iterator end = text_.rend();
    const char* u = utf8.data() + utf8.size();
    int i = 0;
    while (iter != end) {
      char buf[5];
      int n = iter.get_utf8(buf);
      buf[n] = '\0';
      u -= n;
      EXPECT_TRUE(strncmp(buf, u, n) == 0);
      EXPECT_EQ(buf, iter.get_utf8_string());
      EXPECT_EQ(lengths[text_.size() - i - 1], iter.utf8_length());
      iter++;
      i++;
    }
  }

  text_.push_back('$');
  EXPECT_EQ(text_.size(), 6);
  EXPECT_EQ(text_.utf8_length(), 14);

  text_.push_back('\xAE');  // registered sign
  EXPECT_EQ(text_.size(), 7);
  EXPECT_EQ(text_.utf8_length(), 16);  // 2 bytes long
}

TEST_F(IteratorTest, Decode) {
  const char32 text[] = {0x1C0, 0x4E8C, 0xD7DB, 0x34, 0x1D11E};
  UnicodeText::const_iterator iter = text_.begin();
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(text[i], *iter++);
  string s = CodepointString(text_);
  EXPECT_EQ(s, "1C0 4E8C D7DB 34 1D11E ");
}



class OperatorTest : public UnicodeTextTest {};

TEST_F(OperatorTest, Clear) {
  UnicodeText empty_text(UTF8ToUnicodeText(""));
  EXPECT_FALSE(text_ == empty_text);
  text_.clear();
  EXPECT_TRUE(text_ == empty_text);
}

TEST_F(OperatorTest, Empty) {
  EXPECT_TRUE(empty_text_.empty());
  EXPECT_FALSE(text_.empty());
  text_.clear();
  EXPECT_TRUE(text_.empty());
}

TEST(UnicodeTextTest, InterchangeValidity) {
  char* FDD0 = new char[3];
  memcpy(FDD0, "\xEF\xB7\x90", 3);
  EXPECT_FALSE(UniLib::IsInterchangeValid(FDD0, 3));

  UnicodeText a = MakeUnicodeTextWithoutAcceptingOwnership(FDD0, 3);
  EXPECT_EQ(a.size(), 1);
  EXPECT_EQ(*a.begin(), 0x20);
  a.clear();
  a.push_back(0xFDD0);
  EXPECT_EQ(a.size(), 1);
  EXPECT_EQ(*a.begin(), 0x20);

  a = MakeUnicodeTextAcceptingOwnership(FDD0, 3, 3);
  EXPECT_EQ(a.size(), 1);
  EXPECT_EQ(*a.begin(), 0x20);
  a.clear();
  a.push_back(0xFDD0);
  EXPECT_EQ(a.size(), 1);
  EXPECT_EQ(*a.begin(), 0x20);
}

class SubstringSearchTest : public UnicodeTextTest {};

// TEST_F(SubstringSearchTest, FindEmpty) {
//   EXPECT_TRUE(text_.find(empty_text_) == text_.begin());
//   EXPECT_TRUE(empty_text_.find(text_) == empty_text_.end());
// }

// TEST_F(SubstringSearchTest, Find) {
//   UnicodeText::const_iterator second_pos = text_.begin();
//   ++second_pos;
//   UnicodeText::const_iterator third_pos = second_pos;
//   ++third_pos;
//   UnicodeText::const_iterator fourth_pos = third_pos;
//   ++fourth_pos;

//   // same as text_
//   const char32 text[] = {0x1C0, 0x4E8C, 0xD7DB, 0x34, 0x1D11E};

//   UnicodeText prefix;
//   prefix.append(&text[0], &text[2]);
//   EXPECT_TRUE(text_.find(prefix) == text_.begin());
//   EXPECT_TRUE(text_.find(prefix, second_pos) == text_.end());

//   UnicodeText suffix;
//   suffix.append(&text[2], text + arraysize(text));
//   EXPECT_TRUE(text_.find(suffix) == third_pos);
//   EXPECT_TRUE(text_.find(suffix, second_pos) == third_pos);
//   EXPECT_TRUE(text_.find(suffix, third_pos) == third_pos);
//   EXPECT_TRUE(text_.find(suffix, fourth_pos) == text_.end());
// }

// TEST_F(SubstringSearchTest, HasConversionError) {
//   EXPECT_FALSE(text_.HasReplacementChar());
//   const char32 beg[] = {0xFFFD, 0x1C0, 0x4E8C, 0xD7DB, 0x34, 0x1D11E};
//   UnicodeText beg_uni;
//   beg_uni.append(&beg[0], beg + arraysize(beg));
//   EXPECT_TRUE(beg_uni.HasReplacementChar());

//   const char32 mid[] = {0x1C0, 0x4E8C, 0xFFFD, 0xD7DB, 0x34, 0x1D11E};
//   UnicodeText mid_uni;
//   mid_uni.append(&mid[0], mid + arraysize(mid));
//   EXPECT_TRUE(mid_uni.HasReplacementChar());

//   const char32 end[] = {0x1C0, 0x4E8C, 0xD7DB, 0x34, 0x1D11E, 0xFFFD};
//   UnicodeText end_uni;
//   end_uni.append(&end[0], end + arraysize(end));
//   EXPECT_TRUE(end_uni.HasReplacementChar());

//   const char32 two[] = {0xFFFD, 0x1C0, 0x4E8C, 0xD7DB, 0x34, 0x1D11E, 0xFFFD};
//   UnicodeText two_uni;
//   two_uni.append(&two[0], two + arraysize(two));
//   EXPECT_TRUE(two_uni.HasReplacementChar());

//   const char32 adj[] = {0x1C0, 0xFFFD, 0xFFFD, 0x4E8C, 0xD7DB, 0x34, 0x1D11E};
//   UnicodeText adj_uni;
//   adj_uni.append(&adj[0], adj + arraysize(adj));
//   EXPECT_TRUE(adj_uni.HasReplacementChar());
// }

}  // namespace
