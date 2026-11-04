"""calculates the center of a bounding box"""

import tensorflow as tf
import itemgetter
import json
sys.path.insert(0, "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/utils")
from concatenate import concat


annotation_path = "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/COCO2014/annotations/instances_train2014.json"

def calculate_distances():
    """find the distance of the """



def get_regression_target():
    """creates the regression target for the FCOS model"""
    
    boxes = {}
    counter = 0
    
    with open(annotation_path, "r", encoding="utf-8") as f:
        json_obj = json.load(f)
        sorted_imageids = sorted(json_obj["annotations"], key=itemgetter("image_id"))

        for line in sorted_imageids:  
                if counter < 500:                  # due to limited computation had to limit the number of boxes to be loaded
                    if not boxes.get(line["image_id"]):
                        boxes[line["image_id"]] = [concat(line["bbox"], )]
                        
                    elif boxes.get(line["image_id"]):
                        boxes[line["image_id"]] = concat(boxes[line["image_id"]],concat(line["bbox"], line["category_id"]),)
                    
                    counter += 1

    return boxes

