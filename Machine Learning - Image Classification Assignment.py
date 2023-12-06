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

foldersCount = len(next(os.walk('dataset2/triple_mnist/train'))[1])
foldersTrain = next(os.walk('dataset2/triple_mnist/train'))
foldersTest = next(os.walk('dataset2/triple_mnist/test'))

print("Number of Folders:",foldersCount)

#print("Folders List:",foldersTrain)
#path = os.path.basename(os.path.dirname('dataset2/triple_mnist/train/*/*'))

#Reading in Images

Num_Images = glob.glob('dataset2/triple_mnist/train/*/*.png')
Num_Images2 = glob.glob('dataset2/triple_mnist/test/*/*.png')

#Display Image Shape

img_cv2 = cv2.imread(Num_Images[0])
print("Shape:", img_cv2.shape)


#Convert RGB Images to Grayscale

grayImgTrain = []
grayImgTest = []


for i in range(0, int(len(Num_Images))):

    grayImg = (Image.open(Num_Images[i]).convert('L'))
    grayImg = np.array(grayImg) / 255
    grayImgTrain.append(grayImg)

for i in range(0, int(len(Num_Images2))):

    grayImg2 = (Image.open(Num_Images2[i]).convert('L'))
    grayImg2 = np.array(grayImg2) / 255
    grayImgTest.append(grayImg2)

grayImagesTrain = np.array(grayImgTrain)
grayImagesTest = np.array(grayImgTest)

average_shape = np.mean([img.shape for img in grayImagesTrain], axis=0)
print("Average Shape of Images:", average_shape)

#Output Sample Image

fix, ax = plt.subplots(figsize = (8, 8))
ax.axis('off')
ax.set_title(foldersTrain[1][0])
plt.imshow(grayImagesTrain[0], cmap='gray')
plt.show()

#Part 2


vectorImgTrain = []
vectorImgTest = []

#Flatten Images into Vectors

for i in range(0, len(grayImagesTrain)):

    vectorImgTrain.append((grayImagesTrain[i].flatten()))

for i in range(0, len(grayImagesTest)):
    
    vectorImgTest.append((grayImagesTest[i].flatten()))

#print('Min: %.3f, Max: %.3f' % (vectorImgTrain[5].min(), vectorImgTrain[5].max()))

trainImages = np.array(vectorImgTrain)
testImages = np.array(vectorImgTest)

#print("Train Images:", trainImages)
#print("Test Images:", testImages)

y_train = []
y_test = []


print(trainImages.shape)
print(testImages.shape)

for i in range(0, len(foldersTrain[1])):
    for y in range(0, 100):
        y_train.append(int(foldersTrain[1][i]))

for i in range(0, len(foldersTest[1])):
    for y in range(0, 100):  
        y_test.append(int(foldersTest[1][i]))

YTrain = np.array(y_train)
YTest = np.array(y_test)

print("Y_Train:", YTrain)
print("Y_Test:", YTest)

#Logistic Regression Model

# Create and train the Logistic Regression model

model = LogisticRegression(max_iter=1000)
model.fit(trainImages, YTrain)

# Make predictions on the test set
y_pred = model.predict(testImages)

# Evaluate the model
accuracy = metrics.accuracy_score(YTest, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize some test images and their predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(grayImagesTest[i], cmap='gray')  # Use grayImagesTest for visualization
    ax.set_title(f"True: {YTest[i]}, Pred: {y_pred[i]}")
    ax.axis('off')

plt.show()

print("End")

