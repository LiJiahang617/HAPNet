# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv_custom.cnn import Conv2d
from mmcv_custom.ops import point_sample
from mmcv_custom.cnn import build_conv_layer, build_norm_layer
from mmengine_custom.model import ModuleList, caffe2_xavier_init
from mmengine_custom.structures import InstanceData
from torch import Tensor

from mmdet_custom.registry import MODELS, TASK_UTILS
from mmdet_custom.structures import SampleList
from mmdet_custom.utils import ConfigType, OptConfigType, OptMultiConfig, reduce_mean
from ..layers import Mask2FormerTransformerDecoder, SinePositionalEncoding
from ..utils import get_uncertain_point_coords_with_randomness
from ..utils import multi_apply
from .anchor_free_head import AnchorFreeHead
from .maskformer_head import MaskFormerHead


@MODELS.register_module()
class MTMask2FormerHead(MaskFormerHead):
    """Implements the MTMask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`ConfigDict` or dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`ConfigDict` or dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer decoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`ConfigDict` or dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`ConfigDict` or dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            Mask2Former head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            Mask2Former head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 num_bins:int = 256,
                 task_max_val:int = 1,
                 task_min_val:int = 0,
                 task_conv_cfg = dict(type='Conv2d'),
                 num_transformer_feat_level: int = 3,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = dict(
                     num_feats=128, normalize=True),
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=2.0,
                     reduction='mean',
                     class_weight=[1.0] * 133 + [0.1]),
                 loss_mask: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='mean',
                     loss_weight=5.0),
                 loss_dice: ConfigType = dict(
                     type='DiceLoss',
                     use_sigmoid=True,
                     activate=True,
                     reduction='mean',
                     naive_dice=True,
                     eps=1.0,
                     loss_weight=5.0),
                 loss_silog: ConfigType = dict(
                     type='SILogLoss',
                     loss_weight=1.0),
                 loss_bins: ConfigType = dict(
                     type='BinsChamferLoss',
                     loss_weight=0.1),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.layer_cfg. \
            self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding(
            **positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        # for the gen_x head: task aware embedding
        self.ta_glb_reduction = nn.Sequential(nn.Linear(feat_channels, feat_channels//2), nn.ReLU(inplace=True))
        self.ta_glb_embed = nn.Sequential(
            nn.Linear(feat_channels//2 * num_queries, 256), nn.LeakyReLU(), nn.Linear(256, 256),
            nn.LeakyReLU(), nn.Linear(256, num_bins))
        self.ta_mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.LeakyReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.LeakyReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        self.task_max_val = task_max_val
        self.task_min_val = task_min_val
        self.task_conv_out = nn.Sequential(
            build_conv_layer(task_conv_cfg, num_queries, num_bins, kernel_size=1),
            nn.Softmax(dim=1))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)
        # for the gen_x head
        self.loss_silog = MODELS.build(loss_silog)
        self.loss_bins = MODELS.build(loss_bins)

    def init_weights(self) -> None:
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def loss(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        mtl_utils=None
    ) -> Dict[str, Tensor]:
        """Perform forward propagation and loss calculation of the
        multi-task head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            mtl_utils (Tensor): Including input multimodal images
                that may be used by this multi-task learning framework.
                Could be modified by users.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        batch_img_metas = []
        batch_gt_instances = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        # forward multi-task heads. Note: all_pseudo_x_preds is single-channel at that moment
        all_cls_scores, all_mask_preds, all_pseudo_x_preds, all_pseudo_x_bins = self(x, batch_data_samples)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances,
                                                batch_gt_semantic_segs)

        # loss
        losses = self.loss_by_feat(all_cls_scores, all_mask_preds, all_pseudo_x_preds, all_pseudo_x_bins,
                                   batch_gt_instances, batch_img_metas, mtl_utils)

        return losses

    def _get_targets_single(self, cls_score: Tensor, mask_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> Tuple[Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, h, w).
            gt_instances (:obj:`InstanceData`): It contains ``labels`` and
                ``masks``.
            img_meta (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        gt_labels = gt_instances.labels
        gt_masks = gt_instances.masks
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1,
                                                        1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(
            gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1,
                                                               1)).squeeze(1)

        sampled_gt_instances = InstanceData(
            labels=gt_labels, masks=gt_points_masks)
        sampled_pred_instances = InstanceData(
            scores=cls_score, masks=mask_points_pred)
        # assign and sample
        assign_result = self.assigner.assign(
            pred_instances=sampled_pred_instances,
            gt_instances=sampled_gt_instances,
            img_meta=img_meta)
        pred_instances = InstanceData(scores=cls_score, masks=mask_pred)
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds, sampling_result)

    def loss_by_feat(self, all_cls_scores: Tensor, all_mask_preds: Tensor,
                     all_pseudo_x_preds: Tensor, all_pseudo_x_bins: Tensor,
                     batch_gt_instances: List[InstanceData],
                     batch_img_metas: List[dict],
                     mtl_utils=None) -> Dict[str, Tensor]:
        """Loss function. Including segmentation head and gen head

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_list = [
            batch_gt_instances for _ in range(num_dec_layers)
        ]
        img_metas_list = [batch_img_metas for _ in range(num_dec_layers)]
        # seg_head loss
        losses_cls, losses_mask, losses_dice = multi_apply( # 将第一个位置的loss计算方法应用到所有后续位置的超参数中
            self._loss_by_feat_single_seg, all_cls_scores, all_mask_preds,
            batch_gt_instances_list, img_metas_list)
        # gen head loss
        num_items = len(all_pseudo_x_preds)
        mtl_utils_list = [mtl_utils for _ in range(num_items)]
        losses_silog, losses_bins = multi_apply(  # 将第一个位置的loss计算方法应用到所有后续位置的超参数中
            self._loss_by_feat_single_gen, all_pseudo_x_preds, all_pseudo_x_bins, img_metas_list, mtl_utils_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # gen head
        loss_dict['loss_gen_silog'] = losses_silog[-1]
        loss_dict['loss_gen_bins'] = losses_bins[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i, loss_silog_i, loss_bins_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1], losses_silog[:-1], losses_bins[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            # gen head
            loss_dict[f'd{num_dec_layer}.loss_gen_silog'] = loss_silog_i
            loss_dict[f'd{num_dec_layer}.loss_gen_bins'] = loss_bins_i
            num_dec_layer += 1
        return loss_dict

    def _loss_by_feat_single_gen(self, pseudo_x_preds: Tensor, pseudo_x_bins: Tensor,
                             batch_img_metas: List[dict],
                             mtl_utils=None):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            pseudo_x_preds (Tensor): pseudo_x Mask logits for a pixel decoder
                for all images. Shape (batch_size, 1, h//4, w//4) for mask2former
                framework.
            batch_img_metas (list[dict]): List of image meta information.
            mtl_utils (Tensor): Including input multimodal images
                that may be used by this multi-task learning framework.
                Could be modified by users. Shape (batch_size, 4/6, h, w).
            h, w are original image size.
        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        # real_RGB/real_X shape: b,3,h,w
        real_RGB, real_X = torch.split(mtl_utils, (3, 3), dim=1)
        target_shape = pseudo_x_preds.shape[-2:]
        # gt_masks_downsampled shape: b,h//4,w//4
        gt_x_downsampled = F.interpolate(
            real_X[:, 0:1, :, :].float(), target_shape,
            mode='nearest')
        loss_silog = self.loss_silog(
            pseudo_x_preds,
            gt_x_downsampled
            )
        loss_bins = self.loss_bins(
            pseudo_x_bins,
            gt_x_downsampled
        )

        return loss_silog, loss_bins

    def _loss_by_feat_single_seg(self, cls_scores: Tensor, mask_preds: Tensor,
                             batch_gt_instances: List[InstanceData],
                             batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            batch_gt_instances (list[obj:`InstanceData`]): each contains
                ``labels`` and ``masks``.
            batch_img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         avg_factor) = self.get_targets(cls_scores_list, mask_preds_list,
                                        batch_gt_instances, batch_img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([avg_factor]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds.unsqueeze(1), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_preds.unsqueeze(1), points_coords).squeeze(1)

        # dice loss
        loss_dice = self.loss_dice(
            mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_mask, loss_dice

    def _forward_seg_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (batch_size, num_queries, c) -> (batch_size, num_queries, num_classes)
        cls_pred = self.cls_embed(decoder_out)
        # shape (batch_size, num_queries, c) -> (batch_size, num_queries, out_channels)
        mask_embed = self.mask_embed(decoder_out)
        # shape (batch_size, num_queries, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        # reshape to have equal size with feature map
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (batch_size, num_queries, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def _forward_gen_head(self, decoder_out: Tensor, mask_feature: Tensor):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h//4, w//4).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.
            Note: h, w are original image size.
        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h//4, w//4).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h//4, w//4).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (batch_size, num_queries, c) -> (batch_size, num_queries, out_channels)
        ta_mask_embed = self.ta_mask_embed(decoder_out)
        # shape (batch_size, num_queries, h//4, w//4)
        range_attention_maps = torch.einsum('bqc,bchw->bqhw', ta_mask_embed, mask_feature)
        # shape (batch_size, num_queries, c) -> (batch_size, num_queries, c//2)
        ta_glb = self.ta_glb_reduction(decoder_out)
        # shape (batch_size, num_queries, c//2) -> (batch_size, num_queries * c//2)
        ta_glb = ta_glb.flatten(1)
        # shape (batch_size, num_queries * c//2) -> (batch_size, num_bins)
        ta_glb = self.ta_glb_embed(ta_glb)
        ta_glb = torch.relu(ta_glb)
        eps = 0.1
        ta_glb = ta_glb + eps
        # shape = (batch_size, num_bins)
        bin_widths_normed = ta_glb / ta_glb.sum(dim=1, keepdim=True)
        # shape (batch_size, num_queries, h//4, w//4)-> (batch_size, num_bins, h//4, w//4)
        out = self.task_conv_out(range_attention_maps)

        bin_widths = (self.task_max_val - self.task_min_val) * bin_widths_normed
        # shape N, num_bins -> N, num_bins + 1
        bin_widths = F.pad(bin_widths, (1, 0), mode='constant', value=self.task_min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dim_out = centers.size()
        centers = centers.view(n, dim_out, 1, 1)
        # shape (batch_size, num_bins, h, w)-> (batch_size, 1, h, w)
        # TODO: try use 0-1/0-255 to test the difference
        task_pred = torch.sum(out * centers, dim=1, keepdim=True)

        return bin_edges, task_pred

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should include background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_size = len(batch_img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memories (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level): # 3
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            # shape -> (batch_size, h*w, c) c -> 256
            decoder_input = decoder_input + level_embed
            # shape -> (batch_size, h, w)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            # shape -> (batch_size, 2*num_feats, h, w)  num_feats-> 128
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            # shape -> (batch_size, h*w, 2*num_feats)  num_feats-> 128
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))

        cls_pred_list = []
        mask_pred_list = []
        pseudo_x_predict_list = []
        pseudo_x_bins_list = []
        cls_pred, mask_pred, attn_mask = self._forward_seg_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        # TODO: for now, aux head don't need to create new attn_mask
        bin_edges, pseudo_x_pred = self._forward_gen_head(
            query_feat, mask_features)
        # (batch_size, 1, h, w)
        pseudo_x_predict_list.append(pseudo_x_pred)
        # N, num_bins + 1
        pseudo_x_bins_list.append(bin_edges)

        # shape -> (batch_size, num_queries, num_classes)
        cls_pred_list.append(cls_pred) # TODO decide if the first time adding_to_list is necessary
        # shape -> (batch_size, num_queries, h_0, w_0)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers): # 9
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            cls_pred, mask_pred, attn_mask = self._forward_seg_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])
            # for aux head
            bin_edges, pseudo_x_pred = self._forward_gen_head(
                query_feat, mask_features)
            pseudo_x_predict_list.append(pseudo_x_pred)
            # N, num_bins + 1
            pseudo_x_bins_list.append(bin_edges)

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list, pseudo_x_predict_list, pseudo_x_bins_list
