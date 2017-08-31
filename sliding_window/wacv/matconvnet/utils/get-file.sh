#!/bin/bash

local_dir="$1"
url="$2"

function get_filename_from_url() {
regexp='^([^\/]*\/)+'
echo -n "$1" | sed -r "s/$regexp//g"
}

function get_remote_file_size() {
curl -sI "$1" | grep Content-Length | grep -o '[0-9][0-9]*'
}

filename=$(get_filename_from_url "$url")
local_path="$local_dir/$filename"
remote_size=$(get_remote_file_size "$url")

echo "Getting: $url"
echo "     File: $filename"
echo "     Local file path: $local_path"
echo "     Remote file size: $remote_size"

if [ -e "$local_path" ]
then
    local_size=$(stat -c%s "$local_path")
    echo "     Local file size: $local_size"
    if [[ "$local_size" -eq "$remote_size" ]]
    then
        echo "     Local and remote file sizes match: not downloading"
        exit 0
    else
        echo "  Trying to resume partial download"
        if curl -f -C - -o "$local_path" "$url"
        then
            echo "     Download completed successfully"
            exit 0
        else
            echo "     Could not resume"
        fi
    fi
fi

echo "     Downloading the whole file"
curl -f -o "$local_path" "$url"
exit $?
