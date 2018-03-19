/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "syntaxnet/utils.h"
#include "tensorflow/core/platform/macros.h"

namespace syntaxnet {
namespace utils {

bool ParseInt32(const char *c_str, int *value) {
  char *temp;
  *value = strtol(c_str, &temp, 0);  // NOLINT
  return (*temp == '\0');
}

bool ParseInt64(const char *c_str, int64 *value) {
  char *temp;
  *value = strtol(c_str, &temp, 0);  // NOLINT
  return (*temp == '\0');
}

bool ParseDouble(const char *c_str, double *value) {
  char *temp;
  *value = strtod(c_str, &temp);
  return (*temp == '\0');
}

static char hex_char[] = "0123456789abcdef";

string CEscape(const string &src) {
  string dest;

  for (unsigned char c : src) {
    switch (c) {
      case '\n':
        dest.append("\\n");
        break;
      case '\r':
        dest.append("\\r");
        break;
      case '\t':
        dest.append("\\t");
        break;
      case '\"':
        dest.append("\\\"");
        break;
      case '\'':
        dest.append("\\'");
        break;
      case '\\':
        dest.append("\\\\");
        break;
      default:
        // Note that if we emit \xNN and the src character after that is a hex
        // digit then that digit must be escaped too to prevent it being
        // interpreted as part of the character code by C.
        if ((c >= 0x80) || !isprint(c)) {
          dest.append("\\");
          dest.push_back(hex_char[c / 64]);
          dest.push_back(hex_char[(c % 64) / 8]);
          dest.push_back(hex_char[c % 8]);
        } else {
          dest.push_back(c);
          break;
        }
    }
  }

  return dest;
}

std::vector<string> Split(const string &text, char delim) {
  std::vector<string> result;
  int token_start = 0;
  if (!text.empty()) {
    for (size_t i = 0; i < text.size() + 1; i++) {
      if ((i == text.size()) || (text[i] == delim)) {
        result.push_back(string(text.data() + token_start, i - token_start));
        token_start = i + 1;
      }
    }
  }
  return result;
}

std::vector<string> SplitOne(const string &text, char delim) {
  std::vector<string> result;
  size_t split = text.find_first_of(delim);
  result.push_back(text.substr(0, split));
  if (split != string::npos) {
    result.push_back(text.substr(split + 1));
  }
  return result;
}

bool IsAbsolutePath(tensorflow::StringPiece path) {
  return !path.empty() && path[0] == '/';
}

// For an array of paths of length count, append them all together,
// ensuring that the proper path separators are inserted between them.
string JoinPath(std::initializer_list<tensorflow::StringPiece> paths) {
  string result;

  for (tensorflow::StringPiece path : paths) {
    if (path.empty()) {
      continue;
    }

    if (result.empty()) {
      result = path.ToString();
      continue;
    }

    if (result[result.size() - 1] == '/') {
      if (IsAbsolutePath(path)) {
        tensorflow::strings::StrAppend(&result, path.substr(1));
      } else {
        tensorflow::strings::StrAppend(&result, path);
      }
    } else {
      if (IsAbsolutePath(path)) {
        tensorflow::strings::StrAppend(&result, path);
      } else {
        tensorflow::strings::StrAppend(&result, "/", path);
      }
    }
  }

  return result;
}

size_t RemoveLeadingWhitespace(tensorflow::StringPiece *text) {
  size_t count = 0;
  const char *ptr = text->data();
  while (count < text->size() && isspace(*ptr)) {
    count++;
    ptr++;
  }
  text->remove_prefix(count);
  return count;
}

size_t RemoveTrailingWhitespace(tensorflow::StringPiece *text) {
  size_t count = 0;
  const char *ptr = text->data() + text->size() - 1;
  while (count < text->size() && isspace(*ptr)) {
    ++count;
    --ptr;
  }
  text->remove_suffix(count);
  return count;
}

size_t RemoveWhitespaceContext(tensorflow::StringPiece *text) {
  // use RemoveLeadingWhitespace() and RemoveTrailingWhitespace() to do the job
  return RemoveLeadingWhitespace(text) + RemoveTrailingWhitespace(text);
}

namespace {
// Lower-level versions of Get... that read directly from a character buffer
// without any bounds checking.
inline uint32 DecodeFixed32(const char *ptr) {
  return ((static_cast<uint32>(static_cast<unsigned char>(ptr[0]))) |
          (static_cast<uint32>(static_cast<unsigned char>(ptr[1])) << 8) |
          (static_cast<uint32>(static_cast<unsigned char>(ptr[2])) << 16) |
          (static_cast<uint32>(static_cast<unsigned char>(ptr[3])) << 24));
}

// 0xff is in case char is signed.
static inline uint32 ByteAs32(char c) { return static_cast<uint32>(c) & 0xff; }
}  // namespace

uint32 Hash32(const char *data, size_t n, uint32 seed) {
  // 'm' and 'r' are mixing constants generated offline.
  // They're not really 'magic', they just happen to work well.
  const uint32 m = 0x5bd1e995;
  const int r = 24;

  // Initialize the hash to a 'random' value
  uint32 h = seed ^ n;

  // Mix 4 bytes at a time into the hash
  while (n >= 4) {
    uint32 k = DecodeFixed32(data);
    k *= m;
    k ^= k >> r;
    k *= m;
    h *= m;
    h ^= k;
    data += 4;
    n -= 4;
  }

  // Handle the last few bytes of the input array
  switch (n) {
    case 3:
      h ^= ByteAs32(data[2]) << 16;
      TF_FALLTHROUGH_INTENDED;
    case 2:
      h ^= ByteAs32(data[1]) << 8;
      TF_FALLTHROUGH_INTENDED;
    case 1:
      h ^= ByteAs32(data[0]);
      h *= m;
  }

  // Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.
  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;
  return h;
}

string Lowercase(tensorflow::StringPiece s) {
  string result(s.data(), s.size());
  for (char &c : result) {
    c = tolower(c);
  }
  return result;
}

PunctuationUtil::CharacterRange PunctuationUtil::kPunctuation[] = {
    {33, 35},       {37, 42},       {44, 47},       {58, 59},
    {63, 64},       {91, 93},       {95, 95},       {123, 123},
    {125, 125},     {161, 161},     {171, 171},     {183, 183},
    {187, 187},     {191, 191},     {894, 894},     {903, 903},
    {1370, 1375},   {1417, 1418},   {1470, 1470},   {1472, 1472},
    {1475, 1475},   {1478, 1478},   {1523, 1524},   {1548, 1549},
    {1563, 1563},   {1566, 1567},   {1642, 1645},   {1748, 1748},
    {1792, 1805},   {2404, 2405},   {2416, 2416},   {3572, 3572},
    {3663, 3663},   {3674, 3675},   {3844, 3858},   {3898, 3901},
    {3973, 3973},   {4048, 4049},   {4170, 4175},   {4347, 4347},
    {4961, 4968},   {5741, 5742},   {5787, 5788},   {5867, 5869},
    {5941, 5942},   {6100, 6102},   {6104, 6106},   {6144, 6154},
    {6468, 6469},   {6622, 6623},   {6686, 6687},   {8208, 8231},
    {8240, 8259},   {8261, 8273},   {8275, 8286},   {8317, 8318},
    {8333, 8334},   {9001, 9002},   {9140, 9142},   {10088, 10101},
    {10181, 10182}, {10214, 10219}, {10627, 10648}, {10712, 10715},
    {10748, 10749}, {11513, 11516}, {11518, 11519}, {11776, 11799},
    {11804, 11805}, {12289, 12291}, {12296, 12305}, {12308, 12319},
    {12336, 12336}, {12349, 12349}, {12448, 12448}, {12539, 12539},
    {64830, 64831}, {65040, 65049}, {65072, 65106}, {65108, 65121},
    {65123, 65123}, {65128, 65128}, {65130, 65131}, {65281, 65283},
    {65285, 65290}, {65292, 65295}, {65306, 65307}, {65311, 65312},
    {65339, 65341}, {65343, 65343}, {65371, 65371}, {65373, 65373},
    {65375, 65381}, {65792, 65793}, {66463, 66463}, {68176, 68184},
    {-1, -1}};

void NormalizeDigits(string *form) {
  for (size_t i = 0; i < form->size(); ++i) {
    if ((*form)[i] >= '0' && (*form)[i] <= '9') (*form)[i] = '9';
  }
}

}  // namespace utils
}  // namespace syntaxnet
