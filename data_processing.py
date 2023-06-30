import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.utils import shuffle

def process_data(path):
    
    labels=[]
    train_images=[]
    for root, dirs, files in os.walk(path):
        for file in tqdm([f for f in files if f.endswith('.png')]):
            src= root+'/'+file
            label=cv2.imread(src,0)
            label[label>0]=255
            label=cv2.resize(label,(128,128))
            label=label[...,np.newaxis]
            label[label>0]=255
            labels.append(label)
            img=cv2.imread(src[:-3]+"jpg")
            img=cv2.resize(img,(128,128))
            train_images.append(img)
            
    train_images = np.array(train_images)
    labels = np.array(labels)
   
    # Normalize labels - training images get normalized to start in the network
    labels = labels / 255

    # Shuffle images along with their labels, then split into training/validation sets
    train_images, labels = shuffle(train_images, labels, random_state = 123)
    
    return train_images, labels
