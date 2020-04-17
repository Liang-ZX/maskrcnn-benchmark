# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images_left, images_right, targets_left=None, targets_right=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and (targets_left is None or targets_right is None):
            raise ValueError("In training mode, targets should be passed")
        images_left = to_image_list(images_left)
        images_right = to_image_list(images_right)
        features_left = self.backbone(images_left.tensors)
        features_right = self.backbone(images_right.tensors)
        proposals_left, proposals_right, proposal_losses = self.rpn(images_left, images_right, features_left, \
                                features_right, targets_left, targets_right)
        if self.roi_heads:
            x_left, x_right, result_left, result_right, detector_losses = self.roi_heads(features_left, features_right,\
                       proposals_left, proposals_right, targets_left, targets_right)
        else:
            # RPN-only models don't have roi_heads
            x_left = features_left
            x_right = features_right
            result_left = proposals_left
            result_right = proposals_right
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result_left, result_right
