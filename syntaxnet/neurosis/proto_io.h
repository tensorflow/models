#ifndef NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_PROTO_IO_H_
#define NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_PROTO_IO_H_

#include <memory>
#include <string>
#include <vector>

#include "neurosis/utils.h"
#include "neurosis/document_format.h"
#include "neurosis/feature_extractor.pb.h"
#include "neurosis/feature_types.h"
#include "neurosis/registry.h"
#include "neurosis/sentence.pb.h"
#include "task_context.h"
#include "neurosis/workspace.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/io/inputbuffer.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace neurosis {

// A convenience wrapper to read protos with a RecordReader.
class ProtoRecordReader {
 public:
  explicit ProtoRecordReader(tensorflow::RandomAccessFile *file)
      : file_(file), reader_(new tensorflow::io::RecordReader(file_)) {}

  explicit ProtoRecordReader(const string &filename) {
    TF_CHECK_OK(
        tensorflow::Env::Default()->NewRandomAccessFile(filename, &file_));
    reader_.reset(new tensorflow::io::RecordReader(file_));
  }

  ~ProtoRecordReader() {
    reader_.reset();
    delete file_;
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
  tensorflow::RandomAccessFile *file_ = nullptr;
  uint64 offset_ = 0;
  std::unique_ptr<tensorflow::io::RecordReader> reader_;
};

// A convenience wrapper to write protos with a RecordReader.
class ProtoRecordWriter {
 public:
  explicit ProtoRecordWriter(const string &filename) {
    TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file_));
    writer_.reset(new tensorflow::io::RecordWriter(file_));
  }

  ~ProtoRecordWriter() {
    writer_.reset();
    delete file_;
  }

  template <typename T>
  void Write(const T &proto) {
    TF_CHECK_OK(writer_->WriteRecord(proto.SerializeAsString()));
  }

 private:
  tensorflow::WritableFile *file_ = nullptr;
  std::unique_ptr<tensorflow::io::RecordWriter> writer_;
};

// Reads sentence protos from a text conll file.
class TextReader {
 public:
  explicit TextReader(const string &filename)
      : filename_(filename), format_(DocumentFormat::Create("conll-sentence")) {
    Reset();
  }

  Sentence *Read() {
    vector<Sentence *> documents;
    string key, value;
    ReadLines(&key, &value);
    format_->ConvertFromString(key, value, &documents);
    CHECK_LE(documents.size(), 1);
    if (documents.size() == 0) {
      return nullptr;
    } else {
      return documents[0];
    }
  }

  void Reset() {
    line_count_ = 0;
    static const int kInputBufferSize = 1 * 1024 * 1024; /* bytes */
    tensorflow::RandomAccessFile *file;
    TF_CHECK_OK(
        tensorflow::Env::Default()->NewRandomAccessFile(filename_, &file));
    buffer_.reset(new tensorflow::io::InputBuffer(file, kInputBufferSize));
  }

 private:
  void ReadLines(string *key, string *value) {
    string line;
    *key = tensorflow::strings::StrCat(filename_, ":", line_count_);
    value->clear();
    while (buffer_->ReadLine(&line) == tensorflow::Status::OK() &&
           !line.empty()) {
      ++line_count_;
      tensorflow::strings::StrAppend(value, line, "\n");
    }
  }

  string filename_;
  int line_count_ = 0;
  std::unique_ptr<tensorflow::io::InputBuffer> buffer_;
  std::unique_ptr<DocumentFormat> format_;
};

// Writes sentence protos to a text conll file.
class TextWriter {
 public:
  explicit TextWriter(const string &filename)
      : filename_(filename), format_(DocumentFormat::Create("conll-sentence")) {
    TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file_));
  }

  ~TextWriter() {
    file_->Close();
    delete file_;
  }

  void Write(const Sentence &document) {
    string key, value;
    format_->ConvertToString(document, &key, &value);
    TF_CHECK_OK(file_->Append(value));
  }

 private:
  string filename_;
  std::unique_ptr<DocumentFormat> format_;
  tensorflow::WritableFile *file_;
};

}  // namespace neurosis

#endif  // NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_PROTO_IO_H_
