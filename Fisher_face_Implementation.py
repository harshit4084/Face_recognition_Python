import cv2
import os
import numpy as np


print('Loading Training Images...')
training_images = []
training_labels = []

no_of_class = 0
for path in enumerate(os.listdir('orl_faces')):
	join = os.path.join('orl_faces',path[1])
	training_images.append([])
	training_labels.append([])
	training_labels[no_of_class] = path[1]
	for subject in enumerate(os.listdir(join)):
		if subject[1] != '9.pgm':
			training_images[no_of_class].append(os.path.join(join,subject[1]))
	no_of_class += 1
print(' Training Images succefully loaded.')

total_mean = cv2.imread(training_images[0][0],0)
r,c = total_mean.shape
total_mean = np.array(total_mean.flatten('C'),dtype = 'f')


X = []
mean = []
total_image = 0
for class_no in  range(no_of_class):
	X.append([])
	mean.append([])
	mean[class_no] = np.array(cv2.imread(training_images[class_no][0],0).flatten('C'),dtype = 'f')
	for i in range(len(training_images[class_no])):
		img  = cv2.imread(training_images[class_no][i],0) 
		img = np.array(img.flatten('C'),dtype = 'f')
		X[class_no].append(img)
		mean[class_no]+= img    #adding first image twice hence dividing by n+1 in each
		total_mean += img
		total_image+=1
	X[class_no] = np.mat(X[class_no])
	mean[class_no] = np.mat(mean[class_no])/(len(X[class_no])+1)
total_mean = np.mat(total_mean)/(total_image+1)

print(total_mean.shape)
print(mean[0].shape)
print('X bulid succefully. Calculating SVD...')

#####################
# svd code here
A = np.zeros((total_image,r*c),np.float)  # shape is 90x10304
for class_no in range(no_of_class):
	for j in range(len(X[class_no])):
		A[j]= X[class_no][j]-total_mean
# Apply SVD 
u, s, vh = np.linalg.svd(A, full_matrices=False)
print( u.shape, s.shape, vh.shape)
######################
print('Calculating between-class scatter matrix')
SB = np.zeros((r*c,r*c),np.float)
for class_no in range(no_of_class):
	subtract = mean[class_no] - total_mean
	SB += len(X[class_no])*subtract.T@subtract
print(SB.shape)

print('Calculating within-class scatter matrix')
SW = np.zeros((r*c,r*c),np.float)

for class_no in range(no_of_class):
	for j in range(len(X[class_no])):
		subtract = X[class_no][j] - mean[class_no]
		SW += subtract.T@subtract
print(SW.shape)

old_val = 0
w = []
for i in range(20):
	w.append(vh[i])
	W = np.mat(w)
	print('W matrix is ',W.shape)

	numerator = np.linalg.det(W@SB@W.T)
	denomenator = np.linalg.det(W@SW@W.T)
	new_val = numerator/denomenator
	if new_val > old_val:
		print('new_val',new_val)
		old_val = new_val
	

############################
# testing image
print('Loading Testing Images...')
Y = []
for class_no in range(no_of_class):
	Y.append([])
	Y[class_no] = np.mat(W@X[class_no].T)

test_images = []
test_labels = []
for path in enumerate(os.listdir('orl_faces')):
	join = os.path.join('orl_faces',path[1])
	for subject in enumerate(os.listdir(join)):
		if subject[1] == '9.pgm':
			test_images.append(os.path.join(join,subject[1]))
			test_labels.append(join)

t = 9 #set the subject number for testing
while(t>1):
	newimg = cv2.imread(test_images[t-1],0)
	newY = W@np.array(newimg.flatten('C'),dtype = 'f')

	min_dist = 99999.0
	label = 0
	for class_no in range(no_of_class):
		for i in range(len(Y[class_no].T)):
			dist = np.linalg.norm(newY - Y[class_no].T[i])
			#print('Euclidean dist',dist)
			if min_dist < 0.5:
				break
			if dist< min_dist:
				label = class_no
				min_dist = dist

	print('Predicted label = ',training_labels[label])
	print('Actual label = ',test_labels[t-1])
	t-=1


