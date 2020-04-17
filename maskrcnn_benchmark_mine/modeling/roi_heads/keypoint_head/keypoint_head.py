import torch

from .roi_keypoint_feature_extractors import make_roi_keypoint_feature_extractor
from .roi_keypoint_predictors import make_roi_keypoint_predictor
from .inference import make_roi_keypoint_post_processor
from .loss import make_roi_keypoint_loss_evaluator


class ROIKeypointHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIKeypointHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_keypoint_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_keypoint_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_keypoint_post_processor(cfg)
        self.loss_evaluator = make_roi_keypoint_loss_evaluator(cfg)

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
            with torch.no_grad():
                proposals_left, sampled = self.loss_evaluator.subsample(proposals_left, targets_left)
                proposals_right, _ = self.loss_evaluator.subsample(proposals_right, targets_right, sampled)

        x_left = self.feature_extractor(features_left, proposals_left)
        x_right = self.feature_extractor(features_right, proposals_right)
        kp_logits_left = self.predictor(x_left)
        kp_logits_right = self.predictor(x_right)

        if not self.training:
            result_left = self.post_processor(kp_logits_left, proposals_left)
            result_right = self.post_processor(kp_logits_right, proposals_right)
            return x_left, x_right, result_left, result_right, {}

        loss_kp = self.loss_evaluator(proposals_left + proposals_right, kp_logits_left + kp_logits_right)

        return x_left, x_right, proposals_left, proposals_right, dict(loss_kp=loss_kp)


def build_roi_keypoint_head(cfg, in_channels):
    return ROIKeypointHead(cfg, in_channels)
