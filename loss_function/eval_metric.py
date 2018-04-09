import pandas as pd
import numpy as np

import torch
from torch.autograd import Function, Variable

# Since we're extending autograd, our new class
# needs to inherit from `Function`.
class EvaluationMetric(Function):

    def get_iou_vector(self,pred_mask,gt_mask):
        # get intersection and union
        intersection = np.logical_and(pred_mask,gt_mask)
        union = np.logical_or(pred_mask,gt_mask)

        # calculate IoU, which is intersection / union
        iou = np.sum(intersection > 0) / np.sum(union > 0)

        # sweep over range of IoU thresholds and compare IoU
        s = pd.Series()
        for t in np.arange(0.5,1,0.05):
            s[t] = iou > t

        return s

    # we need to get the IoU vector for *each* predicted mask vs. *each*
    # ground truth mask, within an image.
    # num_steps here is 10 because there are 10 steps between 0.5 and 1
    #  when step size is 0.05.
    def get_all_iou_matrices_for_one_image(self,pred_masks,gt_masks,num_steps=10):
        all_iou_mats = np.zeros([num_steps,len(pred_masks),len(gt_masks)])

        # loop over each pred_mask and gt_mask combination, getting the
        # IoU vector. After receiving each vector, add to the dataframe.
        for ii, p in enumerate(pred_masks):
            for jj, g in enumerate(gt_masks):
                s = get_iou_vector(p,g)
                # add the IoU vector to the right part of the total
                #  IoU matrix.
                all_iou_mats[:,ii,jj] = s.values
        
        return all_iou_mats

    # calculate the precision for one IoU matrix,
    # which includes IoU vectors across all thresholds.
    def get_iou_thresh_precision(self,iou_mat):
        
        tp = np.sum(iou_mat.sum(axis=1) > 0)
        fp = np.sum(iou_mat.sum(axis=1) == 0)
        fn = np.sum(iou_mat.sum(axis=0) == 0)
        
        precision = tp / (tp + fp + fn)
        
        return precision

    # get the mean precision for one image.
    def get_mean_precision_for_one_image(self,all_iou_mats):
        all_precisions_for_one_image = []
        
        for thresh, iou_mat in zip(np.arange(0.5,1,0.05), all_iou_mats):
            precision = get_iou_thresh_precision(iou_mat)
            all_precisions_for_one_image.append(precision)

        image_level_mean_precision = np.mean(all_precisions_for_one_image)

        return image_level_mean_precision

    # TODO: Figure out how to input `image_tensors`, or what's appropriate
    #  there. It may be that we need to initialize the number
    #  of images in this class's __init__.
    # get the mean of mean precisions, across all images.
    def get_mean_of_mean_precisions_of_all_images(self,image_tensors,\
        pred_masks,gt_masks):

        im_precisions = []
        for im in image_tensors:
            # `pred_masks` and `gt_masks` are expected to be numpy arrays here.
            all_iou_mats = get_all_iou_matrices_for_one_image(pred_masks,gt_masks)
            mean_precision_for_one_image = get_mean_precision_for_one_image(all_iou_mats)
            im_precisions.append(mean_precision_for_one_image)
        
        # get the mean of all image-level mean precisions.
        np_output = np.mean(im_precisions)

        # convert mean of means from numpy array to tensor.
        output = torch.from_numpy(np_output)

        return output

    # TODO: update inputs appropriately here. input_tensors might need
    #   to be variables. See here:
    #   http://pytorch.org/docs/master/notes/extending.html
    #  Maybe relevant, from documentation:
    #   Variable arguments will be converted to Tensor s before the call,
    #   and their use will be registered in the graph.
    #
    # Both `forward` and `backward` are static methods and
    # require the @staticmethod decorator.
    @staticmethod
    def forward(self,input,target,bias=None,\
        image_tensors,pred_masks,gt_masks):

        self.save_for_backward(input, target, bias=None)

        # this output is of type Tensor
        output = get_mean_of_mean_precisions_of_all_images(image_tensors,\
        pred_masks,gt_masks)

        return output

    # TODO: finish this
    # Both `forward` and `backward` are static methods and
    # require the @staticmethod decorator.
    @staticmethod
    def backward(self,grad_output):
        input, weight, bias = self.saved_tensors
        grad_input = grad_weight = grad_bias = None

    # TODO: This part needs to be updated
    if ctx.needs_input_grad[0]:
        grad_input = grad_output.mm(weight)
    if ctx.needs_input_grad[1]:
        grad_weight = grad_output.t().mm(input)
    if bias is not None and ctx.needs_input_grad[2]:
        grad_bias = grad_output.sum(0).squeeze(0)

    return grad_input, grad_weight, grad_bias