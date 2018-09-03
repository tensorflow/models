import argparse
import cv2
import xml.etree.ElementTree as ET
import os
import glob

# TODO: Enable 2 different scaling parameters (x-axis and y-axis)

def mkdir_if_doesnt_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print('creation of folder {}'.format(path))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='shrink dataset image sizes. '
                                                 'WARNING : IMAGESETS MUST BE CREATED MANUALLY IF NEEDED')
    parser.add_argument('--dataset_path', dest='dataset_path', help='Original dataset path', required=True)
    parser.add_argument('--scale', dest='scale', help='Shrink factor', type=float, required=True)
    #parser.add_argument('--scale_x', dest='scale_x', help='Shrink (x axis) factor)', required=True)
    #parser.add_argument('--scale_y', dest='scale_y', help='Shrink (y axis) factor)', required=True)


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    annotations = os.path.join(args.dataset_path, 'Annotations')
    annotations = os.path.join(annotations, '*.xml')

    imagesdir = os.path.join(args.dataset_path, 'JPEGImages')

    # Constructs output directory path
    model_name = os.path.basename(args.dataset_path)
    parent_path = os.path.abspath(os.path.join(args.dataset_path, '..'))
    save_name = model_name + '_small'
    savedir = os.path.join(parent_path, save_name)

    # Create output directories
    mkdir_if_doesnt_exist(savedir)
    mkdir_if_doesnt_exist(os.path.join(savedir, 'Annotations'))
    mkdir_if_doesnt_exist(os.path.join(savedir, 'JPEGImages'))

    xmls = glob.glob(annotations)
    for xml in xmls:
        print xml
        tree = ET.parse(xml)
        size = tree.find("size")
        size.find("width").text = str(int(int(size.find("width").text)/args.scale))
        size.find("height").text = str(int(int(size.find("height").text)/args.scale))
        for obj in tree.findall("object"):
            bbox = obj.find("bndbox")
            bbox.find("xmax").text = str(int(int(bbox.find("xmax").text)/args.scale))
            bbox.find("xmin").text = str(int(int(bbox.find("xmin").text)/args.scale))
            bbox.find("ymax").text = str(int(int(bbox.find("ymax").text)/args.scale))
            bbox.find("ymin").text = str(int(int(bbox.find("ymin").text)/args.scale))
        tree.write(os.path.join(savedir,"Annotations",os.path.basename(xml)))
        impath = os.path.join(imagesdir,os.path.basename(xml).replace(".xml",".jpg"))
        if os.path.exists(impath):
            image = cv2.imread(impath)
            image = cv2.resize(image, (int(image.shape[1] / args.scale), int(image.shape[0] / args.scale)))
            cv2.imwrite(os.path.join(savedir,"JPEGImages",os.path.basename(impath)),image)
        else:
            print 'image {} not found, ignored'.format(impath)

    print('\n\n...done')
    print('Dataset created here : {}'.format(savedir))
    print('WARNING : IMAGESETS MUST BE CREATED MANUALLY IF NEEDED')
