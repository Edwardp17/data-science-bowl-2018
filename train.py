# TODO: This should be removed eventually.
#  'data' in the paths needs to be changed to '../input/'
# TRAIN_PATH = 'data/stage1_train/'
# TEST_PATH = 'data/stage1_test/'

# TODO: Remove this if the `LoadData` class works correctly.
# Get train and test IDs
# train_ids = next(os.walk(TRAIN_PATH))[1]
# test_ids = next(os.walk(TEST_PATH))[1]

# TODO: Remove when cleaning up.
# X_train, y_train = load_data(ids=train_ids,path=TRAIN_PATH,training_data=True)

import numpy as np
import pandas as pd
import os

from PIL import Image
from skimage.transform import resize
from skimage.io import imshow

import torch

class LoadData():

    def __init__(self,train_path='data/stage1_train/',test_path='data/stage1_test/'):
        self.train_ids = next(os.walk(train_path))[1]
        self.test_ids = next(os.walk(test_path))[1]

    def load_data(self,ids,path,training_data,im_folder='images',mask_folder='masks',im_file_type='.png',im_dim_1=520,im_dim_2=696,im_dim_3=4):
        
        # ids: the array of image ids
        # path: the general directory path, e.g. TRAIN_PATH or TEST_PATH
        # training data: a boolean that specifies whether the data being loaded is training data
        # im_folder: the image subfolder
        # mask_folder: the mask subfolder
        # im_file_type: the image file type. an assumption is made that
        #  the file type is the same for every image.
        # im_dim_1: the first standardization dimension to convert images to.
        # im_dim_2: the second standardization dimension to convert images to.
        # im_dim_3: the third standardization dimension to convert images to.
        
        # initialize empty lists
        X = []
        if training_data: y = []
        
        # print progress
        print(str(len(ids))+' images to load. This may take a while!')
        print('Loading dataset..')

        for i in range(len(ids)):
            # print progress metrics in notebook
            if i % 10 == 0 and i != 0:
                print(str(i)+'/'+str(len(ids))+' images loaded..')
            
            i = ids[i]
            
            # general directory path
            dir_path = path+i
            
            # load image
            im = Image.open(dir_path+'/'+im_folder+'/'+i+im_file_type)
            arr_im = np.asarray(im)
            
            # resize the image to standardize dimensions
            # TODO: Check if `mode` indeed needs to be 'constant' here
            arr_im = resize(arr_im, (im_dim_1,im_dim_2,im_dim_3),mode='constant', preserve_range=True)
            
            t_im = torch.from_numpy(arr_im)
            
            # add numpy image array to X list
            X.append(t_im)
            
            if training_data:

                # TODO: double check that we want np.bool here
                arr_full_mask = np.zeros((im_dim_1,im_dim_2,im_dim_3),dtype=np.bool)

                # BONUS_TODO: [2] in for statement below could be dynamic
                for mask_file in next(os.walk(dir_path+'/'+mask_folder+'/'))[2]:
                    # load a mask
                    im_mask = Image.open(dir_path+'/'+mask_folder+'/'+mask_file)
                    # convert mask from image to array
                    arr_mask = np.asarray(im_mask)
                    # overlay this mask over every other mask for this image.
                    # given the nuclei areas are white and
                    # the areas with no nuclei are black, we can
                    # use np.maximum() here.
                    # first, we standardize the dimensions of the mask so it
                    # fits the image.
                    arr_mask = resize(arr_mask,(im_dim_1,im_dim_2,im_dim_3),mode='constant',preserve_range=True)

                    arr_full_mask = np.maximum(arr_full_mask,arr_mask)

                t_full_mask = torch.from_numpy(arr_full_mask)
                y.append(t_full_mask)
        
        return X, y