import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Function, Variable

class IoU(nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

    # inputs here are both tensors
    def forward(self, pred_mask, gt_mask):
        """
            Args:
                pred_mask: (tensor)
                gt_mask: (tensor)
        """
        intersection = torch.min(pred_mask,gt_mask)
        union = torch.max(pred_mask,gt_mask)

        iou = torch.sum(intersection > 0) / torch.sum(union > 0)

        # since we're building a loss function, we want it to be
        # the opposite of what we're optimizing for.
        return (1 - iou)