import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
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

    names=[]
    # for i in xml_df['filename']:
    #     names.append(i+'.jpg')
    # xml_df['filename']=names

    return xml_df


def main():
    for folder in ['train', 'test']:
        if(folder== 'train'):
            image_path = r'C:\Users\alexs\OneDrive\Documents\GitHub\models\research\images\train'
            xml_df = xml_to_csv(image_path)
            xml_df.to_csv((r'C:\Users\alexs\OneDrive\Documents\GitHub\models\research\images\train_labels.csv'), index=None)
        else:
            image_path = r'C:\Users\alexs\OneDrive\Documents\GitHub\models\research\images\test'
            xml_df = xml_to_csv(image_path)
            xml_df.to_csv((r'C:\Users\alexs\OneDrive\Documents\GitHub\models\research\images\test_labels.csv'), index=None)
        
        
        print('Successfully converted xml to csv.')
main()
