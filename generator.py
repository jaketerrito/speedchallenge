import cv2
import numpy as np
import random
import skimage

'''
A Generator that produces sets of training features and labels
* video:  A list of video frames
* speeds: A list of corresponding speeds
* batch_size: number of samples to produce for each batch
* sequence_length: number of images that model consider for each prediction
'''
def data_generator(video, speeds, batch_size, sequence_length):
	while True:
		sequences = []
		speed_vals = []
		next_frames = []
		while len(sequences) < batch_size:
			frame_num = random.randrange(sequence_length,len(video)-1)
			sequence = video[frame_num-sequence_length:frame_num]
			next_frame = video[frame_num+1]
			'''
			flip = random.choice([True,False])
			angle = random.uniform(-20,20)
			scale = random.uniform(.8,1.2)
			for i, image in enumerate(sequence):
				# Augmentation
				image = skimage.transform.rescale(image, scale=scale)
				image = skimage.transform.resize(image, output_shape=sequence[i].shape, mode='constant')
				image = skimage.transform.rotate(image, angle=angle)

				# Really need to see the types of values before we add this noise
				image = image + np.random.normal(scale=.5,size=image.shape)
				
				sequence[i] = image
			'''
			sequences.append(sequence)
			speed_vals.append(speeds[frame_num])
			next_frames.append(next_frame)
		yield np.array(sequences), [np.array(next_frames), np.array(speed_vals)]

def prediction_generator(video, sequence_length):
	i = 0
	while True:

		yield np.array([video[i:i+sequence_length]])
		i += 1
