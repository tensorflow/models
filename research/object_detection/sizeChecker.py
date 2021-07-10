#!/bin/python3
# -*- encode: utf-8 -*-

"""
This mini program looks for xmin, ymin, xmax and ymax values in order to
check if the figures have the proper size (min 33px)
Version: 0.1
Author: Moisés Lodeiro-Santiago @ https://github.com/moiseslodeiro
"""

from xml.etree import ElementTree
from termcolor import colored
from os.path import isdir
from os import makedirs, rmdir
from sys import exit
from glob import glob as xmlFiles
import shutil
import argparse

# Directories
train_directory = './images/train' # It should contain the xml files with bounding boxes
test_directory = './images/test' # It should contain the xml files with bounding boxes

# Keep calm and do not touch the rest of the code :3
parser = argparse.ArgumentParser()
parser.add_argument("--move", help="Put all wrong xml and images to a wrong_data folder inside each folder", action="store_true")
args = parser.parse_args()

if not isdir(train_directory) or not isdir(test_directory):
   print(colored('[!]', 'yellow', attrs=['bold']), colored('The training or test directories do not exist'))
   exit(1)
else:
    print(colored('[Ok]', 'green'), colored('Directories exists'))

everythingWentAsExpected = True

for tree in [train_directory, test_directory]:
    if args.move and not isdir(tree + '/wrong_data'):
        makedirs(tree + '/wrong_data')

    for file in xmlFiles(tree + '/*.xml'):
       xmlFile = ElementTree.parse(file)
       boxes = xmlFile.findall('object/bndbox')
       for box in boxes:
          xmin, ymin, xmax, ymax = box.getchildren()
          x_value = int(xmax.text) - int(xmin.text)
          y_value = int(ymax.text) - int(ymin.text)
          
          if x_value < 33 or y_value < 33:
             print(colored('[!]', 'red'), 'File {} contains a bounding box smaller than 32 in height or width'.format(file))
             print(colored('xmax - xmin', 'yellow', attrs=['bold']), x_value)
             print(colored('ymax - ymin', 'yellow', attrs=['bold']), y_value)
             everythingWentAsExpected = False
             
             if args.move:
                wrongPicture = xmlFile.find('filename')
                try:
                    shutil.move(file, tree + '/wrong_data/')
                    shutil.move(tree + '/' + wrongPicture.text, tree + '/wrong_data/')
                    print(colored('Files moved to' + tree + '/wrong_data', 'blue'))
                except Exception as e:
                    print(colored(e, 'blue'))

if everythingWentAsExpected:
    print(colored('[Ok]', 'green'), 'All bounding boxes are equal or larger than 32 :-)')
    try:
       rmdir(train_directory + '/wrong_data')
       rmdir(test_directory + '/wrong_data')
    except OSError:
       print(colored('[Info]', 'blue'), 'Directories wrong_data were not removed because they contain some files')

else:
   print()
   print(colored('[Error]', 'red'), ' (╯°□°)╯ ┻━┻')
