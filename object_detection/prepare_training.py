from __future__ import division
import os
from random import choice as rchoice
import pandas as pd
import xml.etree.ElementTree as ET
from generate_tfrecord import generate_from_images

BASEDIR = 'dog-data/'
TRAININGDIR = 'data/'
ALLFILES = 'training-all.txt'

SETS = ( 'train', 'validate' )

def extract_validation():
    ret = {}
    all_files = BASEDIR + ALLFILES
    validation_file = BASEDIR + 'train-validate.txt'
    training_file = BASEDIR + 'train-train.txt'
    if not os.path.exists(validation_file) or not os.path.exists(training_file):
        with open(all_files) as fall:
            all_training = fall.read().splitlines()
        validation_set = []
        while len(validation_set) < 0.1 * len(all_training):
            choice = rchoice(all_training)
            all_training.remove(choice)
            validation_set.append(choice)
        with open(training_file, 'w') as out_file:
            out_file.write('\n'.join(all_training))

        with open(validation_file, 'w') as out_file:
            out_file.write('\n'.join(validation_set))

    else:
        print('using existing training ({}) and validation({})'.format(training_file, validation_file))
    return (training_file, validation_file)

def xml_to_csv(files, basedir):
    xml_list = []
    for xml_name in files:
        xml_file = basedir + xml_name + '.xml'
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def prepare_file(file, dest):
    print('process ' + file)

file_train, file_validate = extract_validation()

labels = [ 'reserved' ]

for set in SETS:
    print('set: {}'.format(set))
    file_train = BASEDIR + 'train-' + set + '.txt'
    with open(file_train) as ftrain:
        trn = ftrain.read().splitlines()
        xml_df = xml_to_csv(trn, BASEDIR)
        #outfile = TRAININGDIR
        xml_df.to_csv(TRAININGDIR + 'train-' + set + '.csv', index=None)
        #file = BASEDIR + 'IMG_0654.JPG'
        print(set, xml_df['class'].count())
        for rec_no in xrange(xml_df['class'].count()):
            rec = xml_df.iloc[rec_no]
            cl_name = rec['class']
            if not cl_name in labels:
                labels.append(cl_name)
        tfrec_file = TRAININGDIR + 'record-' + set + '.record'

        generate_from_images(tfrec_file, xml_df, set, BASEDIR, labels)

print('class names: ', labels)

with open(TRAININGDIR + 'dogs.pbtxt', 'w') as rec_file:
    for id in xrange(len(labels)):
        if id > 0:
            lines = ('item {', '  id: ' + str(id), '  name: ' + labels[id], '}' )
            rec_file.write('\n'.join(lines))
            rec_file.write('\n')


