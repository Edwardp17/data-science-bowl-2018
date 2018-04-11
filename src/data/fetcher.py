import os
import numpy as np

import torch
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# kaggle_data is an unofficial Kaggle data import module.
# from kaggle_data.downloader import KaggleDataDownloader


class DatasetFetcher:
    # renamed our data folder to `input` to standardize to kaggle custom
    # TODO: data_dir can be a more intuitive input. right now it
    # is the directory relative to the script_dir, which is the path
    # to this file.
    # train_folder and test_folder could also probably be more dynamic.
    def __init__(self,\
        competition_name='data-science-bowl-2018',\
        data_dir='../../input/',\
        train_folder='stage1_train',\
        test_folder='stage1_test'):
        """
            TODO: Update description.
            A tool used to automatically download, check, split and get
            relevant information on the dataset
        """
        self.train_data = None
        self.test_data = None
        self.train_masks_data = None
        self.train_files = None
        self.test_files = None
        self.train_masks_files = None
        self.train_ids = None
        self.masks_ids = None
        self.test_ids = None

        # Adding more general attributes
        self.competition_name = competition_name
        self.data_dir = data_dir
        self.train_folder = train_folder
        self.test_folder = test_folder

        # These attributes get populated when we split
        # our training data into training and validation
        # in get_train_files.
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None

        self.X_test = None

    def download_dataset(self):
        """
        TODO: Update description.
        Downloads the dataset and return the input paths
        Args:
            None
        Returns:
            list: [train_data, test_data, metadata_csv, train_masks_csv, train_masks_data]
        """
        competition_name = self.competition_name

        # __file__ here references this current file - so, the first line
        # here that defines script_dir sets script_dir to the
        # path of the current file.
        # TODO: script_dir and destination_path could be more dynamic.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # TODO: This is hacky, update.
        os.chdir(script_dir)
        os.chdir(self.data_dir)
        self.train_data = str(os.getcwd())+'/'+self.train_folder
        self.test_data = str(os.getcwd())+'/'+self.test_folder
        os.chdir(script_dir)
        # destination_path = os.path.join(script_dir, self.data_dir)
        # TODO: This needs to be more dynamic.
        # TODO: not sure why `files` is relevant. take a second look and
        # delete if unnecessary.
        # files = ["stage1_train.zip", "stage1_test.zip", "stage1_train_labels.csv.zip"]
        # NOTE: It's more concise to set train_data and test_data here than below.
        # That way, we can use these values in datasets_path as well.
        # original_folder_path = os.getcwd()
        # self.train_data = os.getcwd()
        
        datasets_path = [self.train_data, self.test_data]
                        # TODO: we don't need "stage1_train_labels.csv" right now. later,
                        # we should implement a more dynamic way of inputting all data
                        # that needs to be retrieved from kaggle.
                        #  destination_path + "stage1_train_labels.csv"]
        
        is_datasets_present = True
        # If the folders already exists then the files may already be extracted
        # This is a bit hacky but it's sufficient for our needs
        for dir_path in datasets_path:
            if not os.path.exists(dir_path):
                is_datasets_present = False

        if not is_datasets_present:
            # NOTE: Updating this to simply print that we need to download Kaggle datasets,
            # given we're not implementing the kaggle_data module yet.
            # Put your Kaggle user name and password in a $KAGGLE_USER and $KAGGLE_PASSWD env vars respectively
            # downloader = KaggleDataDownloader(os.getenv("KAGGLE_USER"), os.getenv("KAGGLE_PASSWD"), competition_name)

            # for file in files:
            #     output_path = downloader.download_dataset(file, destination_path)
            #     downloader.decompress(output_path, destination_path)
            #     os.remove(output_path)
            print("Not all datasets are present. Please download from Kaggle.")
        else:
            print("All datasets are present.")

        # self.train_data = destination_path + train_folder
        # self.test_data = destination_path + test_folder
        self.train_files = sorted(os.listdir(self.train_data))
        self.test_files = sorted(os.listdir(self.test_data))

        # Add validation to make sure that there are 
        # no training or test files that have duplicate ids.
        all_train_ids = [t for t in self.train_files]
        unique_train_ids = list(set(t for t in self.train_files))
        if len(all_train_ids) != len(unique_train_ids):
            raise Exception("There are duplicate training IDs. Please check.")
        self.train_ids = unique_train_ids

        all_test_ids = [t for t in self.test_files]
        unique_test_ids = list(set(t for t in self.test_files))
        if len(all_test_ids) != len(unique_test_ids):
            raise Exception("There are duplicate test IDs. Please check.")
        self.test_ids = unique_test_ids
        
        return datasets_path

    # TODO: sample_size has been removed as a parameter for now, but can be added later.
    # def get_train_files(self, validation_size=0.2, sample_size=None):
    # NOTE: Adding im_folder, mask_folder, and im_file_type here, similar to the way we had it in load_data.
    # NOTE: Dimensions have been changed to convert each image to an RGBA 512x512 square.
    # TODO: Update dimension variable names to be more intuitive.
    def get_train_files(self, validation_size=0.2,im_folder='images',mask_folder='masks',im_file_type='.png',\
        im_dim_1=512,im_dim_2=512,im_dim_3=4):
        """
        Args:
            validation_size (float):
                 Value between 0 and 1
            sample_size (float, None):
                Value between 0 and 1 or None.
                Whether you want to have a sample of your dataset.
        Returns:
            list :
                Returns the dataset in the form:
                [train_data, train_masks_data, valid_data, valid_masks_data]
        """
        # validate that train_ids is not None
        if self.train_ids == None:
            raise Exception("train_ids is None. Do you need to run DatasetFetcher.download_dataset?")
        train_ids = self.train_ids

        # TODO: This has been descoped for now, but can be implemented later.
        # if sample_size:
        #     rnd = np.random.choice(self.train_ids, int(len(self.train_ids) * sample_size))
        #     train_ids = rnd.ravel()

        if validation_size:
            ids_train_split, ids_valid_split = train_test_split(train_ids, test_size=validation_size)
        else:
            ids_train_split = train_ids
            ids_valid_split = []
        
        X_train = []
        y_train = []
        X_valid = []
        y_valid = []

        # TODO: Clean this up.
        dataset_ids = {}
        dataset_ids['X_train'] = ids_train_split
        # datasets[y_train] = ids_train_split
        dataset_ids['X_valid'] = ids_valid_split
        # datasets[y_valid] = ids_valid_split

        for X_name, X, y in zip(['X_train','X_valid'],[X_train,X_valid],[y_train,y_valid]):

            d_ids = dataset_ids[X_name]
            # We use the range() here so that we can easily track progress and print
            # our progress to the user after every 10 images that are loaded.
            for i in range(len(d_ids)):
                # print progress. 
                if i % 10 == 0 and i != 0:
                    print(str(i)+'/'+str(len(d_ids))+' images loaded..')

                # get the actual id of the image.
                id = d_ids[i]

                im = Image.open(self.train_data+'/'+id+'/'+im_folder+'/'+id+im_file_type)
                arr_im = np.asarray(im)

                # resize the image to standardize dimensions
                # TODO: Check if `mode` indeed needs to be 'constant' here
                arr_im = resize(arr_im, (im_dim_1,im_dim_2,im_dim_3),mode='constant', preserve_range=True)
                # convert numpy array to tensor
                t_im = torch.from_numpy(arr_im)
                # append tensor to X
                X.append(t_im)
            
            # TODO: double check that we want np.bool here
            arr_full_mask = np.zeros((im_dim_1,im_dim_2,im_dim_3),dtype=np.bool)

            # BONUS_TODO: [2] in for statement below could be dynamic
            for mask_file in next(os.walk(self.train_data+'/'+id+'/'+mask_folder+'/')):
                # load a mask
                im_mask = Image.open(self.train_data+'/'+id+'/'+mask_folder+'/'+mask_file)
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
            
            # convert numpy array to tensor
            t_full_mask = torch.from_numpy(arr_full_mask)
            # append tensor to y
            y.append(t_full_mask)

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

        # Unlike Ekami's get_train_files, we're returning lists of tensors, not numpy arrays.
        # NOTE: If we change the code downstream, this function doesn't need to return anything.
        # We can get the same information by referencing the DatasetFetcher's relevant attributes.
        # NOTE: the images are resized in get_train_files.,
        return X_train, y_train, X_valid, y_valid

    # TODO: Implement get_test_files
    # def get_test_files(self, sample_size):
    #     test_files = self.test_files

    #     if sample_size:
    #         rnd = np.random.choice(self.test_files, int(len(self.test_files) * sample_size))
    #         test_files = rnd.ravel()

    #     ret = [None] * len(test_files)
    #     for i, file in enumerate(test_files):
    #         ret[i] = self.test_data + "/" + file

    #     return np.array(ret)

    def get_test_files(self,im_folder='images',im_file_type='.png',\
        im_dim_1=512,im_dim_2=512,im_dim_3=4):

        # validate that test_ids is not None
        if self.test_ids == None:
            raise Exception("test_ids is None. Do you need to run DatasetFetcher.download_dataset?")
        test_ids = self.test_ids

        # TODO: This has been descoped for now, but can be implemented later.
        # if sample_size:
        #     rnd = np.random.choice(self.train_ids, int(len(self.train_ids) * sample_size))
        #     train_ids = rnd.ravel()

        X_test = []

        # TODO: Clean this up.
        dataset_ids = {}
        dataset_ids['X_test'] = test_ids

        for X_name, X in zip(['X_test'],[X_test]):

            d_ids = dataset_ids[X_name]
            # We use the range() here so that we can easily track progress and print
            # our progress to the user after every 10 images that are loaded.
            for i in range(len(d_ids)):
                # print progress. 
                if i % 10 == 0 and i != 0:
                    print(str(i)+'/'+str(len(d_ids))+' images loaded..')

                # get the actual id of the image.
                id = d_ids[i]

                im = Image.open(self.test_data+'/'+id+'/'+im_folder+'/'+id+im_file_type)
                arr_im = np.asarray(im)

                # resize the image to standardize dimensions
                # TODO: Check if `mode` indeed needs to be 'constant' here
                arr_im = resize(arr_im, (im_dim_1,im_dim_2,im_dim_3),mode='constant', preserve_range=True)
                # convert numpy array to tensor
                t_im = torch.from_numpy(arr_im)
                # append tensor to X
                X.append(t_im)

        self.X_test = X_test

        # Unlike Ekami's get_train_files, we're returning lists of tensors, not numpy arrays.
        # NOTE: If we change the code downstream, this function doesn't need to return anything.
        # We can get the same information by referencing the DatasetFetcher's relevant attributes.
        # NOTE: the images are resized in get_train_files.,
        return X_test

        #NUTBUTTER APPROVED