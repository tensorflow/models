#!/bin/bash
cd ../../..
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./official/recommendation/data_server/server_command.proto

