import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# kaggle_data is an unofficial Kaggle data import module.
# from kaggle_data.downloader import KaggleDataDownloader


class DatasetFetcher:
    # renamed our data folder to `input` to standardize to kaggle custom
    # TODO: data_dir can be a more intuitive input. right now it
    # is the directory relative to the script_dir, which is the path
    # to this file.
    # train_folder and test_folder could also probably be more dynamic.
    def __init__(self,,\
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

        # Adding more attributes
        self.competition_name = competition_name
        self.data_dir = data_dir
        self.train_folder = train_folder
        self.test_folder = test_folder

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
        destination_path = os.path.join(script_dir, self.data_dir)
        # TODO: This needs to be more dynamic.
        # TODO: not sure why `files` is relevant. take a second look and
        # delete if unnecessary.
        # files = ["stage1_train.zip", "stage1_test.zip", "stage1_train_labels.csv.zip"]
        # NOTE: It's more concise to set train_data and test_data here than below.
        # That way, we can use these values in datasets_path as well.
        self.train_data = destination_path + train_folder
        self.test_data = destination_path + test_folder
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
            raise ValueError("There are duplicate training IDs. Please check.")
        self.train_ids = unique_train_ids

        all_test_ids = [t for t in self.test_files]
        unique_test_ids = list(set(t for t in self.test_files))
        if len(all_test_ids) != len(unique_test_ids):
            raise ValueError("There are duplicate test IDs. Please check.")
        self.test_ids = unique_test_ids
        
        return datasets_path

    #This part below is where I noticed the larger differences in his data vs ours
    def get_nuc_image_files(self, nuc_image_id, test_file=False, get_mask=False):
        if get_mask:
            if nuc_image_id in self.masks_ids:
                return [self.train_masks_data + "/" + s for s in self.train_masks_files if nuc_image_id in s]
            else:
                raise Exception("No mask with this ID found")
        elif test_file:
            if nuc_image_id in self.test_ids:
                return [self.test_data + "/" + s for s in self.test_files if nuc_image_id in s]
        else:
            if nuc_image_id in self.train_ids:
                return [self.train_data + "/" + s for s in self.train_files if nuc_image_id in s]
        raise Exception("No image with this ID found")

    def get_image_matrix(self, image_path):
        img = Image.open(image_path)
        return np.asarray(img, dtype=np.uint8)

    def get_image_size(self, image):
        img = Image.open(image)
        return img.size

    def get_train_files(self, validation_size=0.2, sample_size=None):
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
        train_ids = self.train_ids

        # Each id has 16 images but well...
        if sample_size:
            rnd = np.random.choice(self.train_ids, int(len(self.train_ids) * sample_size))
            train_ids = rnd.ravel()

        if validation_size:
            ids_train_split, ids_valid_split = train_test_split(train_ids, test_size=validation_size)
        else:
            ids_train_split = train_ids
            ids_valid_split = []

        train_ret = []
        train_masks_ret = []
        valid_ret = []
        valid_masks_ret = []

        for id in ids_train_split:
            train_ret.append(self.get_nuc_image_files(id))
            train_masks_ret.append(self.get_nuc_image_files(id, get_mask=True))

        for id in ids_valid_split:
            valid_ret.append(self.get_nuc_image_files(id))
            valid_masks_ret.append(self.get_nuc_image_files(id, get_mask=True))

        return [np.array(train_ret).ravel(), np.array(train_masks_ret).ravel(),
                np.array(valid_ret).ravel(), np.array(valid_masks_ret).ravel()]

    def get_test_files(self, sample_size):
        test_files = self.test_files

        if sample_size:
            rnd = np.random.choice(self.test_files, int(len(self.test_files) * sample_size))
            test_files = rnd.ravel()

        ret = [None] * len(test_files)
        for i, file in enumerate(test_files):
            ret[i] = self.test_data + "/" + file

        return np.array(ret)
