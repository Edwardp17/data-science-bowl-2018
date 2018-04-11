import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader

# custom classes
from data.fetcher import DatasetFetcher
import nn.classifier
import nn.unet_nb as unet
from encoding.encoding import RunLengthEncoding


def main(predict=True):
    # Hyperparameters
    print("Setting hyperparameters..")
    im_height=256
    im_width=256
    # NOTE: im_channels is no longer incorporated in resizing images - images are converted to grayscale during resizing.
    im_channels=3 # setting im_channels to 2 since masks are black and white
    print("im_height: "+str(im_height))
    print("im_width: "+str(im_width))
    print("im_channels: "+str(im_channels))
    # input_img_resize = (512, 512)  # The resize size of the input images of the neural net
    #output_img_resize = (388, 388)  # The resize size of the output images of the neural net
    #batch_size = 3
    epochs = 2
    print("epochs: "+str(epochs))
    thresholds = np.arange(0.5,1,0.05)
    validation_size = 0.2
    # NOTE: Removing sample_size for now to descope, can be implemented later.
    # sample_size = None  # Put 'None' to work on full dataset or a value between 0 and 1

    # -- Optional parameters
    # threads = cpu_count()
    use_cuda = torch.cuda.is_available()
    print("cuda: "+str(use_cuda))
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Download the datasets
    print("Downloading dataset if necessary..")
    ds_fetcher = DatasetFetcher()
    ds_fetcher.download_dataset()

    # Get training and validation data.
    # NOTE: All of this data is returned as tensors, not numpy arrays
    # as was originally the case in Ekami's implementation.
    print("Getting training files..")
    X_train, y_train, X_valid, y_valid = ds_fetcher.get_train_files(validation_size=validation_size,im_dim_3=im_channels)
    print("Training files loaded!")
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
    print("Initializing model..")
    net = unet.UNetOriginal(in_shape)
    print("Initializing classifier..")
    classifier = nn.classifier.DSBowlCLassifier(net,epochs)

    # because X_train and y_train are already tensors from get_train_files,
    # we don't need to transform them anymore. We just need to put them
    # together in the same tuple and then input them into a DataLoader.
    print("Loading training data..")
    train_ds = (X_train, y_train)
    train_loader = DataLoader(dataset=train_ds,batch_size=1,pin_memory=use_cuda)

    # same with validation set as with training set
    print("Loading validation data..")
    valid_ds = (X_valid, y_valid)
    valid_loader = DataLoader(dataset=valid_ds,batch_size=1,pin_memory=use_cuda)

    print("Training on {} samples and validating on {} samples.."
          .format(len(train_loader.dataset), len(valid_loader.dataset)))

    classifier.train(train_loader, valid_loader, epochs)
    print("Done training!")

    # TODO: The prediction section is still not fully complete
    if predict == True:
        print("Loading the test dataset..")
        test_ds, file_names = ds_fetcher.get_test_files(im_dim_3=im_channels)
        test_loader = DataLoader(test_ds,batch_size=1,pin_memory=use_cuda,shuffle=False)

        # Predict and save
        # returns a dict of file names and numpy array prediction masks
        print("Making predictions..")
        files_to_pred_masks = classifier.predict(test_loader, file_names)

        print("Encoding predictions..")
        # TODO: We may need to turn predictions from classifications here,
        # but I don't full understand how that factors in.
        # I'm looking at the "Check RLE" section of this notebook, for
        # reference: https://www.kaggle.com/kmader/nuclei-overview-to-submission    
        encoder = RunLengthEncoding()
        # this returns a pandas dataframe
        df_predictions = encoder.encode_predictions(files_to_pred_masks)
        # output predictions to CSV. Note that the index here is required
        # because that is where the file names show up.
        print("Outputing CSV..")
        df_predictions.to_csv('predictions.csv',index=True)
        print("Done with predictions!")

if __name__ == "__main__":
    main()