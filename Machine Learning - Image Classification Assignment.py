# import the modules
import pandas as pd
import numpy as np
from keras.preprocessing import image
import glob
import cv2
import matplotlib.pylab as plt
import tensorflow as tf
import os
from PIL import Image
import PIL
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import re

foldersCount = len(next(os.walk('dataset2/triple_mnist/train'))[1])
foldersTrain = next(os.walk('dataset2/triple_mnist/train'))
foldersTest = next(os.walk('dataset2/triple_mnist/test'))
foldersVal = next(os.walk('dataset2/triple_mnist/val'))

print("Number of Folders:",foldersCount)

#print("Folders List:",foldersTrain)
#path = os.path.basename(os.path.dirname('dataset2/triple_mnist/train/*/*'))

  #Reading in Images

natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

Num_Images = sorted(glob.glob('dataset2/triple_mnist/train/*/*.png'),key=natsort)
Num_Images2 = sorted(glob.glob('dataset2/triple_mnist/test/*/*.png'),key=natsort)
Num_Images3 = sorted(glob.glob('dataset2/triple_mnist/val/*/*.png'),key=natsort)

print(Num_Images[55])

#Display Image Shape

y_train = []
y_test = []
y_val = []


for i in range(0, len(foldersTrain[1])):
    for y in range(0, 100):
        y_train.append(int(foldersTrain[1][i]))

for i in range(0, len(foldersTest[1])):
    for y in range(0, 100):
        y_test.append(int(foldersTest[1][i]))

for i in range(0, len(foldersVal[1])):
    for y in range(0, 100):
        y_val.append(int(foldersVal[1][i]))

img_cv2 = cv2.imread(Num_Images[0])
print("Shape:", img_cv2.shape)


print(y_train[200])

#Convert RGB Images to Grayscale & Normalize the Data

grayImgTrain = []
grayImgTest = []
grayImgVal = []


for i in range(0, int(len(Num_Images))):

    grayImg = (Image.open(Num_Images[i]).convert('L'))
    grayImg = np.array(grayImg) / 255
    grayImgTrain.append(grayImg)

for i in range(0, int(len(Num_Images2))):

    grayImg2 = (Image.open(Num_Images2[i]).convert('L'))
    grayImg2 = np.array(grayImg2) / 255
    grayImgTest.append(grayImg2)

for i in range(0, int(len(Num_Images3))):

    grayImg3 = (Image.open(Num_Images3[i]).convert('L'))
    grayImg3 = np.array(grayImg3)/ 255
    grayImgVal.append(grayImg3)


grayImagesTrain = np.array(grayImgTrain)
grayImagesTest = np.array(grayImgTest)
grayImagesVal = np.array(grayImgVal)

# Display the first 10 digits in the training set
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(grayImagesTrain[i+200], cmap='gray')
    index_of_one = (y_train[i+200])
    plt.title(f"Label: {index_of_one}")
    plt.axis('off')
plt.show()

average_shape = np.mean([img.shape for img in grayImagesTrain], axis=0)
print("Average Shape of Images:", average_shape)

#Output Sample Image

#fix, ax = plt.subplots(figsize = (8, 8))
#ax.axis('off')
#ax.set_title(foldersTrain[1][1])
#plt.imshow(grayImagesTrain[160], cmap='gray')
#plt.show()

#Part 2


vectorImgTrain = []
vectorImgTest = []
vectorImgVal = []

#Flatten Images into Vectors

for i in range(0, len(grayImagesTrain)):

    vectorImgTrain.append((grayImagesTrain[i].flatten()))

for i in range(0, len(grayImagesTest)):
    
    vectorImgTest.append((grayImagesTest[i].flatten()))

for i in range(0, len(grayImagesVal)):
    
    vectorImgVal.append((grayImagesVal[i].flatten()))

#print('Min: %.3f, Max: %.3f' % (vectorImgTrain[5].min(), vectorImgTrain[5].max()))

trainImages = np.array(vectorImgTrain)
testImages = np.array(vectorImgTest)
valImages = np.array(vectorImgVal)

#print("Train Images:", trainImages)
#print("Test Images:", testImages)


print(trainImages.shape)
print(testImages.shape)

YTrain = np.array(y_train)
YTest = np.array(y_test)
YVal = np.array(y_val)

print("Y_Train:", YTrain)
print(YTrain[55])
print("Y_Test:", YTest)
print(YTest[0])
print("Y_Val:", YVal)
print(YVal[0])

label_encoder = LabelEncoder()

YTrain = label_encoder.fit_transform(YTrain)
YTest = label_encoder.fit_transform(YTest)
YVal = label_encoder.fit_transform(YVal)

#Logistic Regression Model


# Create and train the Logistic Regression model 

model = LogisticRegression(max_iter=1500)
print("Training Start...")
model.fit(trainImages, YTrain)

y_pred_val = model.predict(valImages)

# Evaluate the accuracy of the model
accuracy = accuracy_score(YVal, y_pred_val)
print(f"Accuracy of Val: {accuracy * 100:.2f}%")


# Make predictions on the test set
y_pred = model.predict(testImages)

# Evaluate the model
accuracy = metrics.accuracy_score(YTest, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#print("Precision Score for Val: ",precision_score(y_val, y_pred_val))
#print("Recall Score for Val:    ",recall_score(y_val, y_pred_val))
#print("F1-Score for Val :",f1_score(y_val, y_pred_val))

print("Precision Score: ",precision_score(YTest, y_pred, average='macro'))
print("Recall Score:    ",recall_score(YTest, y_pred, average='macro'))
print("F1-Score :",f1_score(YTest, y_pred, average='macro'))

# Visualize some test images and their predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(grayImagesTest[i], cmap='gray')  # Use grayImagesTest for visualization
    ax.set_title(f"True: {YTest[i]}, Pred: {y_pred[i]}")
    ax.axis('off')

plt.show()

print("End")

