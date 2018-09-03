import argparse
import os
import errno
import glob
import xml.etree.ElementTree as ET
import shutil

parser = argparse.ArgumentParser(description='Generate labelmap txt file')
parser.add_argument('--input', help="Glob to multiclass annotations", required=True)
parser.add_argument('--output', help="Output Annotations folder", required=True)

args = parser.parse_args()

LABELS_TO_DELETE = ['neg']
# LABELS_TO_KEEP = ['dulcolax',
#                   'dulcosoft_yellow_small',
#                   'ibupradoll-400',
#                   'lysopaine',
#                   'lysopaine_miel_citron',
#                   'magnevie-stress-resist',
#                   'novanuit-triple',
#                   'phytoxil',
#                   'phytoxil_sans_sucre',
#                   'toplexil',
#                   'toplexil_sans_sucre']

inputPath = args.input
outputPath = args.output

if os.path.exists(outputPath):
    shutil.rmtree(outputPath)
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

xmls = glob.glob(inputPath)
for xml in xmls:
    print xml
    tree = ET.parse(xml)
    root = tree.getroot()
    for obj in tree.findall("object"):
        # if obj.find("name").text not in LABELS_TO_KEEP:
        if obj.find("name").text in LABELS_TO_DELETE:
            root.remove(obj)
    # if len(tree.findall("object")) == 0:
    #     os.remove(xml)
    else:
        tree.write(os.path.join(outputPath, os.path.basename(xml)))