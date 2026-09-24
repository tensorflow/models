"""function to concatenate the lists of bounding box coordinates of two 
images from COCO dataset into one list"""

    
def concat(list1, list2):
    """a concatination function for the COCO dataset bounding box labels"""
    if isinstance(list1[0], list) and isinstance(list2, list):
        alist = list1
        alist.append(list2)
    elif isinstance(list2, int):
        alist = [list1]
        alist.append(list2)
    return alist


# check if need to put all the boxes of a single image in a list instead of keeping them separate.
# concat([1.08, 187.69, 611.59, 285.84],51)
