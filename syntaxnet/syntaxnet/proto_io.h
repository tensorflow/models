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

#ifndef SYNTAXNET_PROTO_IO_H_
#define SYNTAXNET_PROTO_IO_H_

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "syntaxnet/document_format.h"
#include "syntaxnet/feature_types.h"
#include "syntaxnet/registry.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/utils.h"
#include "syntaxnet/workspace.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/buffered_inputstream.h"
#include "tensorflow/core/lib/io/random_inputstream.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace syntaxnet {

// A convenience wrapper to read protos with a RecordReader.
class ProtoRecordReader {
 public:
  explicit ProtoRecordReader(tensorflow::RandomAccessFile *file) {
    file_.reset(file);
    reader_.reset(new tensorflow::io::RecordReader(file_.get()));
  }

  explicit ProtoRecordReader(const string &filename) {
    TF_CHECK_OK(
        tensorflow::Env::Default()->NewRandomAccessFile(filename, &file_));
    reader_.reset(new tensorflow::io::RecordReader(file_.get()));
  }

  ~ProtoRecordReader() {
    reader_.reset();
  }

  template <typename T>
  tensorflow::Status Read(T *proto) {
    string buffer;
    tensorflow::Status status = reader_->ReadRecord(&offset_, &buffer);
    if (status.ok()) {
      CHECK(proto->ParseFromString(buffer));
      return tensorflow::Status::OK();
    } else {
      return status;
    }
  }

 private:
  uint64 offset_ = 0;
  std::unique_ptr<tensorflow::io::RecordReader> reader_;
  std::unique_ptr<tensorflow::RandomAccessFile> file_;
};

// A convenience wrapper to write protos with a RecordReader.
class ProtoRecordWriter {
 public:
  explicit ProtoRecordWriter(const string &filename) {
    TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file_));
    writer_.reset(new tensorflow::io::RecordWriter(file_.get()));
  }

  ~ProtoRecordWriter() {
    writer_.reset();
    file_.reset();
  }

  template <typename T>
  void Write(const T &proto) {
    TF_CHECK_OK(writer_->WriteRecord(proto.SerializeAsString()));
  }

 private:
  std::unique_ptr<tensorflow::io::RecordWriter> writer_;
  std::unique_ptr<tensorflow::WritableFile> file_;
};

// A file implementation to read from stdin.
class StdIn : public tensorflow::RandomAccessFile {
 public:
  StdIn() {}
  ~StdIn() override {}

  // Reads up to n bytes from standard input.  Returns `OUT_OF_RANGE` if fewer
  // than n bytes were stored in `*result` because of EOF.
  tensorflow::Status Read(uint64 offset, size_t n,
                          tensorflow::StringPiece *result,
                          char *scratch) const override {
    CHECK_EQ(expected_offset_, offset);
    if (!eof_) {
      string line;
      eof_ = !std::getline(std::cin, line);
      buffer_.append(line);
      buffer_.append("\n");
    }
    CopyFromBuffer(std::min(buffer_.size(), n), result, scratch);
    if (eof_) {
      return tensorflow::errors::OutOfRange("End of file reached");
    } else {
      return tensorflow::Status::OK();
    }
  }

 private:
  void CopyFromBuffer(size_t n, tensorflow::StringPiece *result,
                      char *scratch) const {
    memcpy(scratch, buffer_.data(), buffer_.size());
    buffer_ = buffer_.substr(n);
    *result = tensorflow::StringPiece(scratch, n);
    expected_offset_ += n;
  }

  mutable bool eof_ = false;
  mutable int64 expected_offset_ = 0;
  mutable string buffer_;

  TF_DISALLOW_COPY_AND_ASSIGN(StdIn);
};

// Reads sentence protos from a text file.
class TextReader {
 public:
  explicit TextReader(const TaskInput &input, TaskContext *context) {
    CHECK_EQ(input.record_format_size(), 1)
        << "TextReader only supports inputs with one record format: "
        << input.DebugString();
    CHECK_EQ(input.part_size(), 1)
        << "TextReader only supports inputs with one part: "
        << input.DebugString();
    filename_ = TaskContext::InputFile(input);
    format_.reset(DocumentFormat::Create(input.record_format(0)));
    format_->Setup(context);
    Reset();
  }

  Sentence *Read() {
    // Skips emtpy sentences, e.g., blank lines at the beginning of a file or
    // commented out blocks.
    std::vector<Sentence *> sentences;
    string key, value;
    while (sentences.empty() && format_->ReadRecord(buffer_.get(), &value)) {
      key = tensorflow::strings::StrCat(filename_, ":", sentence_count_);
      format_->ConvertFromString(key, value, &sentences);
      CHECK_LE(sentences.size(), 1);
    }
    if (sentences.empty()) {
      // End of file reached.
      return nullptr;
    } else {
      ++sentence_count_;
      return sentences[0];
    }
  }

  void Reset() {
    sentence_count_ = 0;
    if (filename_ == "-") {
      static const int kInputBufferSize = 8 * 1024; /* bytes */
      file_.reset(new StdIn());
      stream_.reset(new tensorflow::io::RandomAccessInputStream(file_.get()));
      buffer_.reset(new tensorflow::io::BufferedInputStream(file_.get(),
                                                            kInputBufferSize));
    } else {
      static const int kInputBufferSize = 1 * 1024 * 1024; /* bytes */
      TF_CHECK_OK(
          tensorflow::Env::Default()->NewRandomAccessFile(filename_, &file_));
      stream_.reset(new tensorflow::io::RandomAccessInputStream(file_.get()));
      buffer_.reset(new tensorflow::io::BufferedInputStream(file_.get(),
                                                            kInputBufferSize));
    }
  }

 private:
  string filename_;
  int sentence_count_ = 0;
  std::unique_ptr<tensorflow::RandomAccessFile>
      file_;  // must outlive buffer_, stream_
  std::unique_ptr<tensorflow::io::RandomAccessInputStream>
      stream_;  // Must outlive buffer_
  std::unique_ptr<tensorflow::io::BufferedInputStream> buffer_;
  std::unique_ptr<DocumentFormat> format_;
};

// Writes sentence protos to a text conll file.
class TextWriter {
 public:
  explicit TextWriter(const TaskInput &input, TaskContext *context) {
    CHECK_EQ(input.record_format_size(), 1)
        << "TextWriter only supports files with one record format: "
        << input.DebugString();
    CHECK_EQ(input.part_size(), 1)
        << "TextWriter only supports files with one part: "
        << input.DebugString();
    filename_ = TaskContext::InputFile(input);
    format_.reset(DocumentFormat::Create(input.record_format(0)));
    format_->Setup(context);
    if (filename_ != "-") {
      TF_CHECK_OK(
          tensorflow::Env::Default()->NewWritableFile(filename_, &file_));
    }
  }

  ~TextWriter() {
    if (file_) {
      TF_CHECK_OK(file_->Close());
    }
  }

  void Write(const Sentence &sentence) {
    string key, value;
    format_->ConvertToString(sentence, &key, &value);
    if (file_) {
      TF_CHECK_OK(file_->Append(value));
    } else {
      std::cout << value;
    }
  }

 private:
  string filename_;
  std::unique_ptr<DocumentFormat> format_;
  std::unique_ptr<tensorflow::WritableFile> file_;
};

}  // namespace syntaxnet

#endif  // SYNTAXNET_PROTO_IO_H_
