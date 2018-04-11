import sys
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import nn.losses as losses_utils

class DSBowlCLassifier:
    def __init__(self, net, max_epochs):
        """
        The classifier for carvana used for training and launching predictions
        Args:
            net (nn.Module): The neural net module containing the definition of your model
            max_epochs (int): The maximum number of epochs on which the model will train
        """
        self.net = net
        self.max_epochs = max_epochs
        self.epoch_counter = 0
        self.use_cuda = torch.cuda.is_available()

    # NOTE: Implement if necessary.
    # def restore_model(self, model_path):
    #     """
    #         Restore a model parameters from the one given in argument
    #     Args:
    #         model_path (str): The path to the model to restore
    #     """
    #     self.net.load_state_dict(torch.load(model_path))

    # NOTE: Given that our model evaluates itself after *each* image,
    # our loss criteria is 1 - (Average Precision at Different Intersection
    # Over Union thresholds), rather than (1 - Mean APADIoU). If the model works
    # to minimize the loss criteria for each image, it should minimize average
    # losses across many images as well.
    def _criterion(self, pred_mask, gt_mask):
        
        print("gt_mask is size: "+str(gt_mask.size()))

        # pred_mask = pred_mask.view(-1,1)
        # gt_mask = gt_mask.view(-1,1)

        # print("gt_mask size after view is "+str(gt_mask.size()))
        # print("pred_mask size after view is "+str(pred_mask.size()))

        print(type(pred_mask),type(gt_mask))
        # NOTE: explicitly calling forward() might not be
        # right here.
        l = losses_utils.IoU().forward(pred_mask,gt_mask)
        return l

    # NOTE: _train_epoch is used for both training and validation,
    # since the operations are exactly the same in our model and
    # just use different data.                
    def _train_epoch(self, train_loader, optimizer):

        all_losses = []

        for ind, (im, gt_mask) in enumerate(train_loader):
            if self.use_cuda:
                    im = im.cuda()
                    gt_mask = gt_mask.cuda()

            # convert the input image and target mask to Variables
            im, mask = Variable(im), Variable(gt_mask)

            # forward
            print(im.size())
            im_dims = len(im.size())
            if im_dims == 3:
                print("image has 3 dimensions")
                im = im.unsqueeze(dim=0)
                im = im.expand(-1,3,-1,-1)
                print("new image size:")
                print(im.size())
                im_dims = len(im.size())
                im = im.type(torch.FloatTensor)
                if self.use_cuda:
                    im = im.cuda()
            elif im_dims == 2:
                print("image has 2 dimensions")
                im = im.unsqueeze(dim=0)
                im = im.unsqueeze(dim=1)
                im = im.expand(-1,3,-1,-1)
                print("new image size:")
                print(im.size())
                im_dims = len(im.size())
                im = im.type(torch.FloatTensor)
                if self.use_cuda:
                    im = im.cuda()

            if im_dims == 4:
                
                sys.stdout.write("\033[0;32m") # green
                print(">>>> STEPPING FORWARD <<<<")
                sys.stdout.write("\033[0;0m") # reset
                pred_mask = self.net(im)
                # NOTE: The immediately below isn't relevant to us
                # because we want our model to output probabilities.
                # probs = F.sigmoid(logits)
                # pred = (probs > threshold).float()

                # backward + optimize
                print("Calculating loss..")
                
                # manage gt_mask variable
                gt_mask = gt_mask.type(torch.FloatTensor)
                if self.use_cuda:
                    gt_mask = gt_mask.cuda()

                loss = self._criterion(pred_mask, Variable(gt_mask))
                optimizer.zero_grad()
                sys.stdout.write("\033[0;32m") # green
                print(">>>> STEPPING BACKWARD <<<<")
                sys.stdout.write("\033[0;0m") # reset
                loss.backward()
                optimizer.step()

                print("Appending loss")
                all_losses.append(loss)

            else:
                print("couldn't get an image to 4 dimensions. not training on it.")

            # TODO: Printing statistics on our performance would be nice.

        print("Returning the mean of all losses..")
        sys.stdout.write("\033[1;34m") # blue
        print(all_losses)
        sys.stdout.write("\033[0;0m") # reset
        return np.mean(all_losses)

    def _run_epoch(self, train_loader: DataLoader, valid_loader: DataLoader,\
        optimizer):

        # switch to train mode
        
        self.net.train()

        print("Training model..")
        train_loss = self._train_epoch(train_loader,optimizer)

        # switch to evaluate mode
        self.net.eval()

        # NOTE: _train_epoch is used for both training and validation,
        # since the operations are exactly the same in our model and
        # just use different data.
        val_loss = self._train_epoch(valid_loader,optimizer)


    def train(self, train_loader: DataLoader, valid_loader: DataLoader,\
        epochs):
        """
            Trains the neural net
        Args:
            train_loader (DataLoader): The Dataloader for training
            valid_loader (DataLoader): The Dataloader for validation
            epochs (int): number of epochs
        Returns:
            str, None: The path where the model was saved, or None if it wasn't saved
        """
        if self.use_cuda:
            self.net.cuda()
            # TODO: Figure out net.parameters here.
        
        optimizer = optim.Adam(self.net.parameters())
        # NOTE: We can implement lr_scheduler later.
        # lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, min_lr=1e-7)

        for epoch in range(epochs):
            print("training epoch "+str(epoch))
            self._run_epoch(train_loader, valid_loader, optimizer)

        # TODO: Double check if we need this.
        # # If there are callback call their __call__ method and pass in some arguments
        # if callbacks:
        #     for cb in callbacks:
        #         cb(step_name="train",
        #            net=self.net,
        #            epoch_id=self.epoch_counter + 1,
        #            )


    def predict(self, test_loader, file_names):
        """
            Launch the prediction on the given loader and pass
            each predictions to the given callbacks.
        Args:
            test_loader (DataLoader): The loader containing the test dataset
        """

        # Switch to evaluation mode
        self.net.eval()

        files_to_pred_masks = {}
        total_img_num = len(test_loader)
        for ind, (im, file_names) in enumerate(zip(test_loader,file_names)):
            
            if self.use_cuda:
                im = im.cuda()

            im = Variable(im)
            
            # forward
            print(im.size())
            im_dims = len(im.size())
            if im_dims == 3:
                print("image has 3 dimensions")
                im = im.unsqueeze(dim=1)
                im = im.expand(-1,3,-1,-1)
                print("new image size:")
                print(im.size())
            elif im_dims == 2:
                print("image has 2 dimensions")
                im = im.unsqueeze(dim=1)
                im = im.unsqueeze(dim=1)
                im = im.expand(-1,3,-1,-1)
                print("new image size:")
                print(im.size())
            
            if im_dims == 4:
                pred_mask = self.net(im)

                # Convert tensor to numpy for return    
                pred_mask = pred_mask.data.cpu().numpy()

                files_to_pred_masks[file_names] = pred_mask
                print("predicted {}/{} image masks".format(ind+1,total_img_num))

            else:

                print("couldn't get test image to 4 dimensions. not predicting")
        
        print(files_to_pred_masks)
        return files_to_pred_masks