import cv2
import os
import numpy as np
import csv

path= r'C:\Users\alexs\OneDrive\Documents\GitHub\models'
os.chdir(path)

cropping = False
 
x_start, y_start, x_end, y_end = 0, 0, 0, 0
 
image = cv2.imread('templateImage.jpg')
#myfile= open('example.csv')
oriImage = image.copy()
 
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
 
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
 
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
 
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
 
        refPoint = [(x_start, y_start), (x_end, y_end)]
 
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            print("x1: ",x_start,"y1: ", y_start," x2: ",x_end,"y2: ", y_end)
            coordinates= [x_start,y_start,x_end,y_end]
            append_list_as_row('example.csv',coordinates)
            cv2.imshow("Cropped", roi)
 
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
 
while True:
 
    i = image.copy()
 
    if not cropping:
        cv2.imshow("image", image)
 
    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)
 
    cv2.waitKey(1)
 
# close all open windows
cv2.destroyAllWindows()