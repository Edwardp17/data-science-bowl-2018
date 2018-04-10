import numpy as np
import os
import torch
from torch.utils.data import DataLoader

# custom classes
from data.fetcher import DatasetFetcher
import nn.unet_nb as unet


def main():
    # Hyperparameters
    im_height=512
    im_width=512
    im_channels=4
    # input_img_resize = (512, 512)  # The resize size of the input images of the neural net
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

    # Download the datasets
    ds_fetcher = DatasetFetcher()
    ds_fetcher.download_dataset()

    # Get training and validation data.
    # NOTE: All of this data is returned as tensors, not numpy arrays
    # as was originally the case in Ekami's implementation.
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(validation_size=validation_size)
    
    # TODO: Implement get_test_files
    # full_x_test = ds_fetcher.get_test_files(sample_size)

    # NOTE: None of this should be relevant to us because we've received our
    # images when loading our data.
    # # Get the original images size (assuming they are all the same size)
    # origin_img_size = ds_fetcher.get_image_size(X_train[0])
    # # The image kept its aspect ratio so we need to recalculate the img size for the nn
    # img_resize_centercrop = transformer.get_center_crop_size(X_train[0], img_resize)  # Training callbacks

    # NOTE: These all look like logs, so we don't need to implement them.
    # tb_viz_cb = TensorboardVisualizerCallback(os.path.join(script_dir, '../logs/tb_viz'))
    # tb_logs_cb = TensorboardLoggerCallback(os.path.join(script_dir, '../logs/tb_logs'))
    # model_saver_cb = ModelSaverCallback(os.path.join(script_dir, '../output/models/model_' +
    #                                                  helpers.get_model_timestamp()), verbose=True)


    # TODO: Double check if this is relevant to making predictions.
    # Testing callbacks
    # pred_saver_cb = PredictionsSaverCallback(os.path.join(script_dir, '../output/submit.csv.gz'),
    #                                          origin_img_size, threshold)


    # Define our neural net architecture.
    in_shape = (im_height, im_width, im_channels)
    net = unet.UNetOriginal(in_shape)

    
    classifier = nn.classifier.CarvanaClassifier(net, epochs)