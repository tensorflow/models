#!/bin/bash
#
# Convenience script to build wheel files in Docker, and copy them out of the
# container.
#
# Usage: docker-devel/build_wheels.sh (takes no arguments; run it from the base
# directory).
set -e
docker build -t dragnn-oss .

# Start building the wheels.
script="bazel run //dragnn/tools:build_pip_package \
  -- --output-dir=/opt/tensorflow/syntaxnet; \
  bazel run //dragnn/tools:build_pip_package \
  -- --output-dir=/opt/tensorflow/syntaxnet --include-tensorflow"
container_id="$(docker run -d dragnn-oss /bin/bash -c "${script}")"

echo "Waiting for container ${container_id} to finish building the wheel ..."
if [[ "$(docker wait "${container_id}")" != 0 ]]; then
  echo "Container failed! Please run \`docker logs <id>\` to see errors." >&2
  exit 1
fi

# The build_pip_package.py script prints lines like "Wrote x.whl". The wheel
# names are prefixed by architecture and such, so don't guess them.
wheels=(
  $(docker logs "${container_id}" 2>/dev/null | grep Wrote | awk '{print $2;}'))
for wheel in "${wheels[@]}"; do
  output=./"$(basename "${wheel}")"
  docker cp "${container_id}:${wheel}" "${output}"
  echo "Wrote ${output} ($(du -h "${output}" | awk '{print $1;}'))"
done

echo "Removing ${container_id} ..."
docker rm "${container_id}" >/dev/null
