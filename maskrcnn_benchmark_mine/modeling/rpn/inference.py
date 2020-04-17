# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes

from ..utils import cat
from .utils import permute_and_flatten

class RPNPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RPN boxes, before feeding the
    proposals to the heads
    """

    def __init__(
        self,
        pre_nms_top_n,
        post_nms_top_n,
        nms_thresh,
        min_size,
        box_coder=None,
        fpn_post_nms_top_n=None,
        fpn_post_nms_per_batch=True,
    ):
        """
        Arguments:
            pre_nms_top_n (int)
            post_nms_top_n (int)
            nms_thresh (float)
            min_size (int)
            box_coder (BoxCoder)
            fpn_post_nms_top_n (int)
        """
        super(RPNPostProcessor, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.box_coder = box_coder

        if fpn_post_nms_top_n is None:
            fpn_post_nms_top_n = post_nms_top_n
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.fpn_post_nms_per_batch = fpn_post_nms_per_batch

    def add_gt_proposals(self, proposals, targets):
        """
        Arguments:
            proposals: list[BoxList]
            targets: list[BoxList]
        """
        # Get the device we're operating on
        device = proposals[0].bbox.device

        gt_boxes = [target.copy_with_fields([]) for target in targets]

        # later cat of bbox requires all fields to be present for all bbox
        # so we need to add a dummy for objectness that's missing
        for gt_box in gt_boxes:
            gt_box.add_field("objectness", torch.ones(len(gt_box), device=device))

        proposals = [
            cat_boxlist((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def forward_for_single_feature_map(self, anchors_left, anchors_right, objectness_left, objectness_right,\
                                box_regression_left, box_regression_right):
        """
        Arguments:
            anchors: list[BoxList]
            objectness: tensor of size N, A, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = objectness_left.device
        N, A, H, W = objectness_left.shape

        # put in the same format as anchors
        objectness_left = permute_and_flatten(objectness_left, N, A, 1, H, W).view(N, -1)
        objectness_right = permute_and_flatten(objectness_right, N, A, 1, H, W).view(N, -1)
        objectness_left = objectness_left.sigmoid()
        objectness_right = objectness_right.sigmoid()

        box_regression_left = permute_and_flatten(box_regression_left, N, A, 4, H, W)
        box_regression_right = permute_and_flatten(box_regression_right, N, A, 4, H, W)

        num_anchors = A * H * W

        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        objectness_left, topk_idx_left = objectness_left.topk(pre_nms_top_n, dim=1, sorted=True)
        objectness_right, topk_idx_right = objectness_right.topk(pre_nms_top_n, dim=1, sorted=True)

        batch_idx = torch.arange(N, device=device)[:, None]
        box_regression_left = box_regression_left[batch_idx, topk_idx_left]
        box_regression_right = box_regression_right[batch_idx, topk_idx_right]

        image_shapes = [box.size for box in anchors_left]
        concat_anchors_left = torch.cat([a.bbox for a in anchors_left], dim=0)
        concat_anchors_left = concat_anchors_left.reshape(N, -1, 4)[batch_idx, topk_idx_left]
        
        concat_anchors_right = torch.cat([a.bbox for a in anchors_right], dim=0)
        concat_anchors_right = concat_anchors_right.reshape(N, -1, 4)[batch_idx, topk_idx_right]

        proposals_left = self.box_coder.decode(
            box_regression_left.view(-1, 4), concat_anchors_left.view(-1, 4)
        )
        proposals_right = self.box_coder.decode(
            box_regression_right.view(-1, 4), concat_anchors_right.view(-1, 4)
        )

        proposals_left = proposals_left.view(N, -1, 4)
        proposals_right = proposals_right.view(N, -1, 4)

        result_left = []; result_right = []
        for proposal_left, score_left, proposal_right, score_right, im_shape in zip(proposals_left, objectness_left,\
                         proposals_right, objectness_right, image_shapes):
            boxlist_left = BoxList(proposal_left, im_shape, mode="xyxy")
            boxlist_right = BoxList(proposal_right, im_shape, mode="xyxy")
            boxlist_left.add_field("objectness", score_left)
            boxlist_right.add_field("objectness", score_right)
            boxlist_left = boxlist_left.clip_to_image(remove_empty=False)
            boxlist_right = boxlist_right.clip_to_image(remove_empty=False)
            boxlist_left = remove_small_boxes(boxlist_left, self.min_size)
            boxlist_right = remove_small_boxes(boxlist_right, self.min_size)
            boxlist_left, boxlist_right = boxlist_nms(
                boxlist_left, boxlist_right,
                self.nms_thresh,
                max_proposals=self.post_nms_top_n,
                score_field="objectness",
            )
            result_left.append(boxlist_left)
            result_right.append(boxlist_right)
        return result_left, result_right

    def forward(self, anchors_left, anchors_right, objectness_left, objectness_right, box_regression_left,\
                box_regression_right, targets_left=None, targets_right=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            objectness: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes_left = []; sampled_boxes_right = []
        num_levels = len(objectness_left) #FPN层数
        anchors_left = list(zip(*anchors_left))
        anchors_right = list(zip(*anchors_right))
        for aleft, aright, oleft, oright, bleft, bright in zip(anchors_left, anchors_right, objectness_left, \
                 objectness_right, box_regression_left, box_regression_right):
            sample_left, sample_right = self.forward_for_single_feature_map(aleft, aright, oleft, oright, bleft, bright)
            sampled_boxes_left.append(sample_left)
            sampled_boxes_right.append(sample_right)

        boxlists_left = list(zip(*sampled_boxes_left))
        boxlists_right = list(zip(*sampled_boxes_right))
        boxlists_left = [cat_boxlist(boxlist_left) for boxlist_left in boxlists_left]
        boxlists_right = [cat_boxlist(boxlist_right) for boxlist_right in boxlists_right]

        if num_levels > 1:
            boxlists_left = self.select_over_all_levels(boxlists_left)
            boxlists_right = self.select_over_all_levels(boxlists_right)

        # append ground-truth bboxes to proposals
        if self.training and targets_left is not None and targets_right is not None:
            boxlists_left = self.add_gt_proposals(boxlists_left, targets_left)
            boxlists_right = self.add_gt_proposals(boxlists_right, targets_right)

        return boxlists_left, boxlists_right

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        # different behavior during training and during testing:
        # during training, post_nms_top_n is over *all* the proposals combined, while
        # during testing, it is over the proposals for each image
        # NOTE: it should be per image, and not per batch. However, to be consistent 
        # with Detectron, the default is per batch (see Issue #672)
        if self.training and self.fpn_post_nms_per_batch:
            objectness = torch.cat(
                [boxlist.get_field("objectness") for boxlist in boxlists], dim=0
            )
            box_sizes = [len(boxlist) for boxlist in boxlists]
            post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
            _, inds_sorted = torch.topk(objectness, post_nms_top_n, dim=0, sorted=True)
            inds_mask = torch.zeros_like(objectness, dtype=torch.uint8) #torch.bool
            inds_mask[inds_sorted] = 1
            inds_mask = inds_mask.split(box_sizes)
            for i in range(num_images):
                boxlists[i] = boxlists[i][inds_mask[i]]
        else:
            for i in range(num_images):
                objectness = boxlists[i].get_field("objectness")
                post_nms_top_n = min(self.fpn_post_nms_top_n, len(objectness))
                _, inds_sorted = torch.topk(
                    objectness, post_nms_top_n, dim=0, sorted=True
                )
                boxlists[i] = boxlists[i][inds_sorted]
        return boxlists


def make_rpn_postprocessor(config, rpn_box_coder, is_train):
    fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN
    if not is_train:
        fpn_post_nms_top_n = config.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST

    pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN
    post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TRAIN
    if not is_train:
        pre_nms_top_n = config.MODEL.RPN.PRE_NMS_TOP_N_TEST
        post_nms_top_n = config.MODEL.RPN.POST_NMS_TOP_N_TEST
    fpn_post_nms_per_batch = config.MODEL.RPN.FPN_POST_NMS_PER_BATCH
    nms_thresh = config.MODEL.RPN.NMS_THRESH
    min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RPNPostProcessor(
        pre_nms_top_n=pre_nms_top_n,
        post_nms_top_n=post_nms_top_n,
        nms_thresh=nms_thresh,
        min_size=min_size,
        box_coder=rpn_box_coder,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        fpn_post_nms_per_batch=fpn_post_nms_per_batch,
    )
    return box_selector
