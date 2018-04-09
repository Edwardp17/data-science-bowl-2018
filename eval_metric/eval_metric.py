# Calculation of evaluation metric adapted with much appreciation from
# this notebook: https://www.kaggle.com/stkbailey/step-by-step-explanation-of-scoring-metric/notebook

import pandas as pd
import numpy as np

def get_iou_vector(pred_mask,gt_mask):
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
def get_all_iou_matrices_for_one_image(pred_masks,gt_masks,num_steps=10):
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

def get_iou_thresh_precision(iou_mat):
    
    tp = np.sum(iou_mat.sum(axis=1) > 0)
    fp = np.sum(iou_mat.sum(axis=1) == 0)
    fn = np.sum(iou_mat.sum(axis=0) == 0)
    
    precision = tp / (tp + fp + fn)
    
    return precision

def get_mean_precision_for_one_image():
    all_precisions_for_one_image = []
    
    for thresh, iou_mat in zip(np.arange(0.5,1,0.05), all_iou_mats):
        precision = get_iou_thresh_precision(iou_mat)
        all_precisions_for_one_image.append(precision)

    mean_precision_for_one_image = np.mean(all_precisions_for_one_image)

    return mean_precision_for_one_image

def get_mean_of_mean_precisions_of_all_images(image_level_mean_precisions):

    mean_of_means = np.mean(image_level_mean_precisions)

    return mean_of_means