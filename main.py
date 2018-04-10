import numpy as np
import torch
from torch.utils.data import DataLoader


def main():
    # Hyperparameters
    #input_img_resize = (572, 572)  # The resize size of the input images of the neural net
    #output_img_resize = (388, 388)  # The resize size of the output images of the neural net
    #batch_size = 3
    epochs = 50
    threshold = np.arange(0.5,1,0.05)
    validation_size = 0.2
    sample_size = None  # Put 'None' to work on full dataset or a value between 0 and 1

    # -- Optional parameters
    threads = cpu_count()
    use_cuda = torch.cuda.is_available()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    #TODO: Training callbacks
    #tb_viz_cb = TensorboardVisualizerCallback(os.path.join(script_dir, '../logs/tb_viz'))
    #tb_logs_cb = TensorboardLoggerCallback(os.path.join(script_dir, '../logs/tb_logs'))
    #model_saver_cb = ModelSaverCallback(os.path.join(script_dir, '../output/models/model_' +
                                                     #helpers.get_model_timestamp()), verbose=True)

    # Download the datasets
    ds_fetcher = DatasetFetcher()
    ds_fetcher.download_dataset()
