import xml.etree.ElementTree as ET
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='Transform xml annotation files')
parser.add_argument('--source', help="Folder containing the xml annotation files", required=True)

args = parser.parse_args()


source = os.path.abspath(args.source)
annotations = os.path.join(source, '*.xml')
destination = os.path.join(source, "../Annotations_tf")
os.mkdir(destination)
xmls = glob.glob(annotations)
for xml in xmls:
    tree = ET.parse(xml)
    fil = tree.find("filename")
    fil.text = fil.text.split('.')[0]
    for obj in tree.findall("object"):
        obj.find("name").text = os.path.basename(xml).split(".")[0]
        lab = "_".join((obj.find("name").text.split('_')[:-1]))
        obj.find("name").text = lab
    tree.write(os.path.join(destination,os.path.basename(xml)))
