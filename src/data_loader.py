import cv2
from tensorflow.keras.utils import Sequence
import numpy as np
from tqdm import tqdm
import os
from config import SIZE

# Convert from RGB to Lab
"""
The DataGenerator2D reads, resizes each image into the given dimensions and then spilts it into train and test set
"""

class DataGenerator2D(Sequence):
    def __init__(self, base_path_color, base_path_bw, img_size=SIZE,batch_size=1, shuffle=True):
        self.base_path_color = base_path_color
        self.base_path_bw = base_path_bw
        self.img_size = img_size        
        self.id = os.listdir(base_path_color)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.ceil(len(self.id)/float(self.batch_size)))

    def __load(self, id_name):
        color_img = cv2.imread(self.base_path_color + '/'+ id_name,1)
        bw_img = cv2.imread(self.base_path_bw + '/'+ id_name,1)
        # open cv reads images in BGR format so we have to convert it to RGB
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        #resizing image
        color_img = cv2.resize(color_img, (self.img_size, self.img_size))
        color_img = color_img.astype('float32') / 255.0
        
        bw_img = cv2.resize(bw_img, (self.img_size, self.img_size))
        bw_img = bw_img.astype('float32') / 255.0
        color_img = np.array(color_img)
        bw_img = np.array(bw_img)
        return bw_img, color_img

    def getitem(self, train_files):
        bw_images, colour_images = [], []
        i=0
        for id_name in tqdm(self.id):
            if i == 6000:
                break
            
            bw_img, color_img = self.load__(id_name)
            bw_images.append(bw_img)
            colour_images.append(color_img)
            i+=1
            
        gray_img = np.array(bw_images)
        color_img = np.array(colour_images)
        train_gray_image = gray_img[:train_files]
        train_color_image = color_img[:train_files]

        test_gray_image = gray_img[train_files:]
        test_color_image = color_img[train_files:]
        
        train_g = np.reshape(train_gray_image,(len(train_gray_image),self.img_size,self.img_size,3))
        train_c = np.reshape(train_color_image, (len(train_color_image),self.img_size,self.img_size,3))

        test_gray_image = np.reshape(test_gray_image,(len(test_gray_image),self.img_size,self.img_size,3))
        test_color_image = np.reshape(test_color_image, (len(test_color_image),self.img_size,self.img_size,3))

        bw_images = np.array(bw_images)
        colour_images  = np.array(colour_images)
        return (train_g, train_c, test_gray_image, test_color_image)


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.id))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



