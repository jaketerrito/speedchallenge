import matplotlib.pyplot as plt
from generator import data_generator, prediction_generator
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU, PReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, RMSprop
from opticalflow import denseflow
from os.path import splitext
import numpy as np
import cv2 as cv

batch_size = 16
sequence_length = 8
epochs = 25
split = .9

print("Processing speeds.")
with open('data/train.txt') as f:
	speeds = f.readlines()
	speeds = np.array([float(x.strip()) for x in speeds])
	speeds = speeds



print("Processing video.")
fname = "data/train.mp4"
try:
	f, ext = splitext(fname)
	with np.load(f + '_op.npz') as data:
		dense_video = data['arr_0']
except:
	print("Could not find preprocessed video, creating it now")
	dense_video = denseflow(fname, 4)

width = dense_video.shape[2]
height = dense_video.shape[1]
video_size = len(dense_video)

model2 = load_model(filepath='./data/weights.hdf5')
model2.compile(RMSprop(),loss='mean_squared_error')

pred_gen = prediction_generator(dense_video, sequence_length)

predictions = model2.predict_generator(pred_gen, steps=500)#video_size-sequence_length)

# Plotting predicted speeds against real speeds
plt.plot(predictions)
plt.plot(speeds)
plt.xlabel('Frame')
plt.ylabel('Speed in mph')
plt.legend(['Predicted', 'Real'])
plt.savefig(fname='./data/speedplot')
plt.show()



cap = cv.VideoCapture(fname)
if (cap.isOpened()== False): 
	print("Error opening video stream or file")
	exit()

count = 0

font = cv.FONT_HERSHEY_SIMPLEX

while(1):
	ret, frame = cap.read()
	if not ret:
		break
	prediction = str(round(predictions[count][0], 1))
	cv.putText(frame,'predicted: ' + prediction,(10,30),font, 1,(255,255,255),2,cv.LINE_AA)
	cv.putText(frame,'real speed: ' + str(speeds[count]),(10,70),font, 1,(255,255,255),2,cv.LINE_AA)
	cv.imshow('frame',frame)
	k = cv.waitKey(30) & 0xff
	if k == 27:
		break
	count += 1
cap.release()