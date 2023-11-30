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



foldersCount = len(next(os.walk('dataset2/triple_mnist/train'))[1])
foldersList = next(os.walk('dataset2/triple_mnist/train'))
print("Number of Folders:",foldersCount)

path = os.path.basename(os.path.dirname('dataset2/triple_mnist/train/*/*'))

#Reading in Images

Num_Images = glob.glob('dataset2/triple_mnist/train/*/*.png')

#Display Image Shape

img_cv2 = cv2.imread(Num_Images[0])
print("Shape:", img_cv2.shape)

grayImg = []


for i in range(0, len(Num_Images)):

    grayImg.append(Image.open(Num_Images[i]).convert('L'))


grayImages = np.array(grayImg)

print("Shape of the grayscale images array:", grayImages.shape)

fix, ax = plt.subplots(figsize = (8, 8))
ax.axis('off')
ax.set_title(foldersList[1][0])
plt.imshow(grayImg[0], cmap='gray')
plt.show()

vectorImg = []

#Flatten Images into Vectors

for i in range(0, len(Num_Images)):

    vectorImg.append(grayImages[i].flatten())

vectorImages = np.array(vectorImg)

print(vectorImages.shape)

print("End")


