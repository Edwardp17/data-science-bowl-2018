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

        # _assert_no_grad(target)
        
        intersection = torch.min(pred_mask,gt_mask)
        union = torch.max(pred_mask,gt_mask)

        # intersection = intersection.type(torch.LongTensor)
        # union = union.type(torch.LongTensor)

        # print(torch.sum(intersection.data > 0))
        # print(torch.sum(union.data > 0))
        # intersection = torch.sum(intersection > 0)
        
        # print(union)
        # union = union > 0
        # print(union)
        # print(union.sum())
        # # intersection = Variable(torch.min(pred_mask.data.cpu(),gt_mask.data.cpu()))
        # # union = Variable(torch.max(pred_mask.data.cpu(),gt_mask.data.cpu()))
        # intersection = torch.clamp(pred_mask,max=gt_mask)
        # union = torch.clamp(pred_mask,min=gt_mask)
        # intersection = np.min(pred_mask.cpu().numpy(),gt_mask.cpu().numpy())
        # union = np.min(pred_mask.cpu().numpy(),gt_mask.cpu().numpy())

        # print("intersection: "+str(torch.sum(intersection > 0.0)))
        # print("union: "+str(torch.sum(union > 0.0)))
        # print("intersection: "+str(torch.sum(intersection)))
        # print("union: "+str(torch.sum(union)))

        # conversion to get to binary using torch 
        intersection = torch.clamp(intersection,min=0,max=1)
        intersection = torch.ceil(intersection)
        intersection = intersection.sum()

        union = torch.clamp(union,min=0,max=1)
        union = torch.ceil(union)
        union = union.sum()

        print(intersection.size())
        print(union.size())

        iou = intersection.div(union)
        print(iou)

        loss = 1 - iou
        # iou = torch.div(torch.FloatTensor(torch.sum(intersection.data > 0)),torch.FloatTensor(torch.sum(union.data > 0)))

        # since we're building a loss function, we want it to be
        # the opposite of what we're optimizing for.
        # print(type(iou))
        # print(iou)
        return loss