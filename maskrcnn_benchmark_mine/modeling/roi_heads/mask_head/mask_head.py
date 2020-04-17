# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator


def keep_only_positive_boxes(boxes_left, boxes_right):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes_left, (list, tuple))
    assert isinstance(boxes_right, (list, tuple))
    assert isinstance(boxes_left[0], BoxList)
    assert isinstance(boxes_right[0], BoxList)
    assert boxes_left[0].has_field("labels")
    assert boxes_right[0].has_field("labels")
    positive_boxes_left = []; positive_boxes_right = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image_left, boxes_per_image_right in zip(boxes_left, boxes_right):
        labels_left = boxes_per_image_left.get_field("labels")
        labels_right = boxes_per_image_right.get_field("labels")
        inds_mask = labels_left > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes_left.append(boxes_per_image_left[inds])
        positive_boxes_right.append(boxes_per_image_right[inds])
        positive_inds.append(inds_mask)
    return positive_boxes_left, positive_boxes_right, positive_inds


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features_left, features_right, proposals_left, proposals_right, targets_left=None, targets_right=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals_left = proposals_left
            all_proposals_right = proposals_right
            proposals_left, proposals_right, positive_inds = keep_only_positive_boxes(proposals_left, proposals_right)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x_left = features_left; x_right = features_right
            x_left = x_left[torch.cat(positive_inds, dim=0)]
            x_right = x_right[torch.cat(positive_inds, dim=0)]
        else:
            x_left = self.feature_extractor(features_left, proposals_left)
            x_right = self.feature_extractor(features_right, proposals_right)
        mask_logits_left = self.predictor(x_left)
        mask_logits_right = self.predictor(x_right)

        if not self.training:
            result_left = self.post_processor(mask_logits_left, proposals_left)
            result_right = self.post_processor(mask_logits_right, proposals_right)
            return x_left, x_right, result_left, result_right, {}

        loss_mask = self.loss_evaluator(proposals_left + proposals_right, mask_logits_left + mask_logits_right, targets_left + targets_right)

        return x_left, x_right, all_proposals_left, all_proposals_right, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)
