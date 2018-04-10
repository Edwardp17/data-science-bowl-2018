
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
        l = losses_utils.IoU().forward(pred_mask,gt_mask)
        return l

    def _train_epoch(self, train_loader, optimizer, threshold):
        for ind, (im, gt_mask) in enumerate(train_loader):
            if self.use_cuda:
                    im = im.cuda()
                    gt_mask = gt_mask.cuda()

            # convert the input image and target mask to Variables
            im, mask = Variable(im), Variable(gt_mask)

            # forward
            pred_masks = self.net.forward(im)
            # NOTE: The immediately below isn't relevant to us
            # because we want our model to output probabilities.
            # probs = F.sigmoid(logits)
            # pred = (probs > threshold).float()

            # backward + optimize
            loss = self._criterion(pred_mask, gt_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # TODO: Printing statistics on our performance would be nice.

            


    def _run_epoch(self, train_loader: DataLoader, valid_loader: DataLoader,\
        optimizer):

        # TODO: Double check if this is relevant
        # switch to train mode
        self.net.train()


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


    def predict():
    