
cd /workspaces/delf && protoc delf/protos/*.proto --python_out=.
protoc delf/protos/*.proto --python_out=.
pip3 --disable-pip-version-check install -e .
