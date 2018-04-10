
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
        pred_mask = pred_mask.view(1,-1)
        gt_mask = gt_mask.view(1,-1)
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
            pred_masks = self.net(im)
            # NOTE: The immediately below isn't relevant to us
            # because we want our model to output probabilities.
            # probs = F.sigmoid(logits)
            # pred = (probs > threshold).float()

            # backward + optimize
            loss = self._criterion(pred_mask, gt_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_losses.append(loss)

            # TODO: Printing statistics on our performance would be nice.

        return np.mean(all_losses)

    def _run_epoch(self, train_loader: DataLoader, valid_loader: DataLoader,\
        optimizer):

        # switch to train mode
        self.net.train()

        train_loss = self._train_epoch(train_loader,optimizer)

        # switch to evaluate mode
        self.net.eval()

        # NOTE: _train_epoch is used for both training and validation,
        # since the operations are exactly the same in our model and
        # just use different data.
        val_loss = self._train_epoch(valid_loader)


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
            # TODO: Make sure net has cuda.
            self.net.cuda()
            # TODO: Figure out net.parameters here.
            optimizer = optim.Adam(self.net.parameters())
            # NOTE: We can implement lr_scheduler later.
            # lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, min_lr=1e-7)

            for epoch in range(epochs):
                self._run_epoch(train_loader, valid_loader, optimizer, lr_scheduler, threshold, callbacks)

        # TODO: Double check if we need this.
        # # If there are callback call their __call__ method and pass in some arguments
        # if callbacks:
        #     for cb in callbacks:
        #         cb(step_name="train",
        #            net=self.net,
        #            epoch_id=self.epoch_counter + 1,
        #            )


    def predict(self, test_loader):
        """
            Launch the prediction on the given loader and pass
            each predictions to the given callbacks.
        Args:
            test_loader (DataLoader): The loader containing the test dataset
        """

        # Switch to evaluation mode
        self.net.eval()

        # pred_masks = []
        files_to_pred_masks = {}
        for ind, (im, file_names) in enumerate(test_loader):

            if self.use_cuda:
                im = im.cuda()

            im = Variable(im)

            # forward
            # forward
            pred_mask = self.net(im)

            # Convert tensor to numpy for return    
            pred_mask = pred_mask.numpy()

            files_to_pred_masks[files_names] = pred_mask
        
        return files_to_pred_masks