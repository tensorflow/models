import xml.etree.cElementTree as ET
from random import randrange
import os

class GenerateXml(object):
    def __init__(self, box_array, im_width, im_height, inferred_class, filename):
        self.inferred_class = inferred_class
        self.box_array = box_array
        self.im_width = im_width
        self.im_height = im_height
        self.file_name = filename

    def get_file_name(self):
        xml_path = './xml'
        directory = os.path.basename(xml_path)
        file_list = os.listdir(directory)

        if len(file_list) == 0:
            return 1
        else:
            return len(file_list) + 1

    def gerenate_basic_structure(self):
        count = 0
        file_name = "" + "" + str(self.file_name)
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "filename").text = file_name + ".jpg"
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(self.im_width)
        ET.SubElement(size, "height").text = str(self.im_height)
        ET.SubElement(size, "depth").text = "3"
        for i in self.box_array:
            objectBox = ET.SubElement(annotation, "object")
            ET.SubElement(objectBox, "name").text = str(self.inferred_class[count])
            print("COUNT : " + str(count) + " CLASS : " + str(self.inferred_class[count]))
            ET.SubElement(objectBox, "pose").text = "Unspecified"
            ET.SubElement(objectBox, "truncated").text = "0"
            ET.SubElement(objectBox, "difficult").text = "0"
            count += 1
            bndBox = ET.SubElement(objectBox, "bndbox")
            ET.SubElement(bndBox, "xmin").text = str(int(i['xmin']))
            ET.SubElement(bndBox, "ymin").text = str(int(i['ymin']))
            ET.SubElement(bndBox, "xmax").text = str(int(i['xmax']))
            ET.SubElement(bndBox, "ymax").text = str(int(i['ymax']))
            
        arquivo = ET.ElementTree(annotation)
        arquivo.write("./xml/" + file_name + ".xml")

def main():
    xml = GenerateXml([{'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}, {'xmin': 0.5406094193458557, 'xmax': 0.6001364588737488, 'ymin': 0.6876631379127502, 'ymax': 0.7547240853309631}], '4000', '2000', 'miner') # just for debuggind
    xml.gerenate_basic_structure()    