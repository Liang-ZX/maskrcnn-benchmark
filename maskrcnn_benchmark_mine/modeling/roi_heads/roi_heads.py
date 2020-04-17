# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features_left, features_right, proposals_left, proposals_right, targets_left=None, targets_right=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x_left, x_right, detections_left, detections_right, loss_box = self.box(features_left, features_right,\
                     proposals_left, proposals_right, targets_left, targets_right)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features_left = features_left
            mask_features_right = features_right
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features_left = x_left
                mask_features_right = x_right
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x_left, x_right, detections_left, detections_right, loss_mask = self.mask(mask_features_left,\
                       mask_features_right, detections_left, detections_right, targets_left, targets_right)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features_left = features_left
            keypoint_features_right = features_right
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features_left = x_left
                keypoint_features_right = x_right
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x_left, x_right, detections_left, detections_right, loss_keypoint = self.keypoint(keypoint_features_left, \
                       detections_left, detections_right, targets_left, targets_right)
            losses.update(loss_keypoint)
        return x_left, x_right, detections_left, detections_right, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
