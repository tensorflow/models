import os
import xml.etree.ElementTree as ET
import sys

def append_jpg_extension(directory):
    # appends ".jpg" to all of the xml files inside the given directory at data["filename"]
    for filename in os.listdir(directory):
        if os.path.splitext(filename)[1] == ".xml":
            tree = ET.parse(os.path.join(directory, filename))
            root = tree.getroot()
            filename_text = root[1].text
            if os.path.splitext(filename_text)[1] != ".jpg":
                root[1].text = filename_text + ".jpg"
                tree.write(os.path.join(directory, filename))

def main():
    append_jpg_extension(sys.argv[1])

if __name__ == "__main__":
    main()
