import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from torch.autograd import Function, Variable

class IoU(nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

        self.use_cuda = torch.cuda.is_available()

    # inputs here are both tensors
    def forward(self, pred_mask, gt_mask):
        """
            Args:
                pred_mask: (Tensor)
                gt_mask: (Tensor)
        """
        
        intersection = torch.min(pred_mask,gt_mask)
        union = torch.max(pred_mask,gt_mask)
        # # intersection = Variable(torch.min(pred_mask.data.cpu(),gt_mask.data.cpu()))
        # # union = Variable(torch.max(pred_mask.data.cpu(),gt_mask.data.cpu()))
        # intersection = torch.clamp(pred_mask,max=gt_mask)
        # union = torch.clamp(pred_mask,min=gt_mask)
        # intersection = np.min(pred_mask.cpu().numpy(),gt_mask.cpu().numpy())
        # union = np.min(pred_mask.cpu().numpy(),gt_mask.cpu().numpy())

        iou = torch.sum(intersection > 0) / torch.sum(union > 0)

        # since we're building a loss function, we want it to be
        # the opposite of what we're optimizing for.
        return (1 - iou)