import numpy as np
from PIL import Image
import os
from skimage.morphology import label

class RunLengthEncoding():
    def __init__(self):
        self.dict_encoded = {}
        # self.df_encoded = None

    def encode_predictions(self, files_to_pred_masks):
        for file_name in files_to_pred_masks.values():
            self.dict_encoded[file_name] = self.rle_encoding(file_name)

        df_encoded = pd.DataFrame.from_dict(dict_encoded,orient='index')

        return df_encoded

    # referenced from here: https://www.kaggle.com/rakhlin/fast-run-length-encoding-python    
    def rle_encoding(self, x):
        '''
        x: numpy array of shape (height, width), 1 - mask, 0 - background
        Returns run length as list
        '''
        dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1): run_lengths.extend((b+1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths