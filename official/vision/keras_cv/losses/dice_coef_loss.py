import tensorflow.keras.backend as K
import numpy as np

def  dice_loss(y_true,y_pred,smooth=1):
	'''
	dice coefficient =2*sum(|y_true*y_pred|)/(sum(y_true^2)+sum(y_pred^2))
	

	'''
	intersection=K.sum(K.abs(y_true*y_pred),axis=-1)
	return 1-(2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) - intersection + smooth)

