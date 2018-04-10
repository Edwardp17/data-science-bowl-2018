import numpy as np
import torch
from torch.utils.data import DataLoader

# custom classes
from data.fetcher import DatasetFetcher


def main():
    # Hyperparameters
    input_img_resize = (512, 512)  # The resize size of the input images of the neural net
    #output_img_resize = (388, 388)  # The resize size of the output images of the neural net
    #batch_size = 3
    epochs = 50
    thresholds = np.arange(0.5,1,0.05)
    validation_size = 0.2
    # NOTE: Removing sample_size for now to descope, can be implemented later.
    # sample_size = None  # Put 'None' to work on full dataset or a value between 0 and 1

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

    # Get the path to the files for the neural net
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(validation_size=validation_size)
    
    # TODO: Implement get_test_files
    # full_x_test = ds_fetcher.get_test_files(sample_size)