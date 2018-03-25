import cv2
import os
import numpy as np

training_images = []
training_labels = []
for path in enumerate(os.listdir('orl_faces')):
	join = os.path.join('orl_faces',path[1])
	for subject in enumerate(os.listdir(join)):
		if subject[1] != '9.pgm':
			training_images.append(os.path.join(join,subject[1]))
			training_labels.append(join)
print('Images succefully loaded')

mean_vector = cv2.imread(training_images[0],0)
r,c = mean_vector.shape
mean_vector = np.array(mean_vector.flatten('C'),dtype = 'f')
training_data = []
for i in range(len(training_images)):
	img  = cv2.imread(training_images[i],0)   # Read the image
	training_data.append(np.array(img.flatten('C'),dtype = 'f')) #convert to coloumn vector and add to list
	mean_vector += np.array(img.flatten('C'),dtype = 'f')
training_data = np.mat(training_data)
#print(training_data)
N = len(training_data) # N is 90
mean_vector = np.mat(mean_vector)/N
M = len(mean_vector.T) # M is 10304
#print(mean_vector)
print('training_data bulid succefully. Calculating SVD...')

A = np.zeros((N,M),np.float)  # shape is 90x10304
for i in range(N):
	A[i]= training_data[i]-mean_vector
# Apply SVD 
u, s, vh = np.linalg.svd(A, full_matrices=False)
#print( u.shape, s.shape, vh.shape)

# Scatter Matrix
scatter = np.matmul(np.mat(A.T),np.mat(A))#np.mat(A.T)@np.mat(A)
#print(scatter.shape)
old_det = 0.0
threshold = 1000000.0
w = []
#print('v shape',vh[0].shape)
for i in range(N):
	w.append(vh[i])
	W = np.mat(w)
	print('W matrix is ',W.shape)
	new_det = np.linalg.det(W@scatter@W.T)
	#print('new_det', new_det)
	#print('old_det', old_det)
	#if (new_det-old_det)<threshold:
	if i>10:
		break
	else:
		old_det = new_det
#print(W.shape)


# testing image
print('testing...')
Y = W@training_data.T
#print('Y shape',Y.shape)
test_images = []
test_labels = []
for path in enumerate(os.listdir('orl_faces')):
	join = os.path.join('orl_faces',path[1])
	for subject in enumerate(os.listdir(join)):
		if subject[1] == '9.pgm':
			test_images.append(os.path.join(join,subject[1]))
			test_labels.append(join)

t = 2 #set the subject number
newimg = cv2.imread(test_images[t-1],0)
#cv2.imshow('newimg',newimg)
#cv2.waitKey(0)
newY = W@np.array(newimg.flatten('C'),dtype = 'f')

#from scipy.spatial.distance import cdist
#dist =cdist(newY, Y, 'euclidean')[0]

min_dist = 99999.0
label = 0
for i in range(len(Y.T)):
	dist = np.linalg.norm(newY - Y.T[i])
	#print('Euclidean dist',dist)
	if min_dist < 0.5:
		break
	if dist< min_dist:
		label = i
		min_dist = dist

matched = training_data[label].reshape(c,r)
np.array(matched,dtype ='uint8')
#cv2.imshow('matched',matched)
#cv2.waitKey(0)
print('Predicted label = ',training_labels[label])
print('Actual label = ',test_labels[t-1])
