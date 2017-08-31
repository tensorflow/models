import os
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

IMAGE_DEPTH = 1
LABEL_FILE = "ground_truth.txt"
DATA_DIR = os.path.expanduser("~/Data/GDXray/Castings")
rootdir = Path(DATA_DIR)


def to_edges(box):
	x1 = int(box[1])
	x2 = int(box[2])
	y1 = int(box[3])
	y2 = int(box[4])

	y = int(box[2])
	w = int(box[3])
	h = int(box[4])
	return [(x1,y1), (x2, y2)]


class Image:

	def __init__(self,pixels,boxes,filename=None):
		self.pixels = pixels
		self.width = self.pixels.shape[1]
		self.height = self.pixels.shape[0]
		self.depth = IMAGE_DEPTH
		self.filename = filename
		self.boxes = boxes

	def draw(self):
		for row in range(self.boxes.shape[0]):
			edges = to_edges(self.boxes[row,:])
			cv.rectangle(self.pixels, edges[0], edges[1], (0,0,255),2)
		self.pixels = self.pixels[:,:,::-1]
		plt.imshow(self.pixels)
		plt.show()


	def dilate(self, width, height):
		"""Dilate width and height to increase box size"""
		sx = width/self.width
		sy = height/self.height
		self.width = width
		self.height = height
		self.pixels = cv.resize(self.pixels, (width,height))
		# Scale boxes too
		self.boxes[:,1] *= sx
		self.boxes[:,2] *= sx
		self.boxes[:,3] *= sy
		self.boxes[:,4] *= sy


	def crop(self, x, y, width, height):
		"""Crop the image and discard extra boxes"""
		self.boxes[:,1] -= x
		self.boxes[:,2] -= x
		self.boxes[:,3] -= y
		self.boxes[:,4] -= y

		newboxes = []
		for row in range(self.boxes.shape[0]):
			box = self.boxes[row,:]
			if box[1]>0 and box[2]<width and box[3]>0 and box[4]<height:
				newboxes.append(box)
		if newboxes:
			self.boxes = np.stack(newboxes)
		else:
			self.boxes = self.boxes[0:0,0]
		self.pixels = self.pixels[y: y + height, x: x + width]
		self.width = width
		self.height = height

	def clone(self, i):
		"""Return a copy of this object"""
		pixels = np.copy(self.pixels)
		boxes = np.copy(self.boxes)
		return Image(pixels, boxes, self.filename)


def crop_and_dilate(image):
		"""Return a series of new images where we zoom in on a defect and dilate the image"""
		Ws = 200 # Sample width
		Hs = 200 # Sample height
		Wt = 300 # Target width
		Ht = 300 # Target height

		newboxes = []
		for row in range(image.boxes.shape[0]):
			box = image.boxes[row,:]
			x = box[1]-random.randint(0,Ws/2)
			y = box[3]-random.randint(0,Hs/2)
			# Enforce crop box
			x = min(max(x,0), image.width-Ws)
			y = min(max(y,0), image.height-Hs)
			# Clone the box and manipulate it
			clone = image.clone(i=row)
			clone.crop(x, y, Ws, Hs)
			if len(clone.boxes):
				clone.dilate(Wt, Ht)
				newboxes.append(clone)
		return newboxes


def get_images():
	"""Iterator for image objects"""

	for folder in rootdir.glob('C*'):
		if not folder.is_dir():
			continue
		label_file = folder/LABEL_FILE

		if label_file.exists():
			labels = np.loadtxt(str(label_file))
		else:
			continue

		for i,filename in enumerate(folder.glob('*.png')):
			index = i+1
			boxes = labels[labels[:,0] == index]
			pixels = cv.imread(str(filename))
			image = Image(pixels,boxes,str(filename))
			yield image

