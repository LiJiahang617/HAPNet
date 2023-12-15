# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional
import inspect
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine_custom.model import BaseModel, is_model_wrapper
from mmengine_custom.structures import PixelData

from mmseg_custom.structures import SegDataSample
from mmseg_custom.registry import MODELS
from mmseg_custom.utils import (ForwardResults, ConfigType, OptConfigType, OptMultiConfig,
                                OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from ..utils import resize


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class CustomModule(nn.Module):
    # 将已有实例封装到一个容器中，方便优化器分配参数组 这里面的name不影响原来容器中的名称
    def __init__(self, module_dict):
        super(CustomModule, self).__init__()
        for name, module in module_dict.items():
            self.add_module(name, module)


@MODELS.register_module()
class MTEncoderDecoder(BaseSegmentor):
    """Multi-task Encoder Decoder architecture for segmentors.

    Multi-task EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSampel`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 generators: OptConfigType = None,
                 discriminator: OptConfigType = None,
                 gen_x_pixel_loss_weight = 1.0,
                 gen_x_gan_loss_weight = 0.01,
                 gen_rgb_pixel_loss_weight = 1.0,
                 gen_rgb_gan_loss_weight=0.01,
                 default_domain_RGB: str = 'X',
                 default_domain_X: str = 'RGB',
                 reachable_domains: List[str] = ['RGB', 'X'],
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if generators is not None:
            self.generators = MODELS.build(generators)
        # used by discriminators
        self._default_domain_R = default_domain_RGB
        self._default_domain_X = default_domain_X
        self._reachable_domains = reachable_domains
        self.gen_x_pixel_loss_weight = gen_x_pixel_loss_weight
        self.gen_rgb_pixel_loss_weight = gen_rgb_pixel_loss_weight
        self.gen_x_gan_loss_weight = gen_x_gan_loss_weight
        self.gen_rgb_gan_loss_weight = gen_rgb_gan_loss_weight
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head) # generator
        self._init_discriminators(discriminator) # discriminator

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        # 使用总模型的子模块构建 CustomModule 实例
        main_head_module_dict = {'decode_head': self.decode_head, 'generators': self.generators}
        self.main_head = CustomModule(main_head_module_dict)

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def _init_discriminators(self, discriminator: ConfigType) -> None:
        # build domain discriminators
        if discriminator is not None:
            self.discriminators = nn.ModuleDict()
            for domain in self._reachable_domains:
                self.discriminators[domain] = MODELS.build(discriminator)
        # support no discriminator in testing
        else:
            self.discriminators = None

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    def _generator_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList):
        """
        Run generator forward function to produce fake images of target domains in
        training.
        """
        # losses = dict()
        gen_outputs = dict()
        if isinstance(self.generators, nn.ModuleList):
            for idx, generator in enumerate(self.generators):
                # TBD: Any generator head need to re-implement the ``forward`` method
                output_gen = generator(inputs, data_samples)
                # losses.update(add_prefix(loss_gen, f'gen_{idx}'))
                gen_outputs.update(add_prefix(output_gen, f'gen_out_{idx}'))
        else:
            output_gen = self.generators(inputs)
            # losses.update(add_prefix(loss_gen, 'gen'))
            gen_outputs.update(add_prefix(output_gen, f'gen_out'))

        return gen_outputs

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict], data_samples) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.
        customed by Kobe Li, add GAN-generator outputs dict
        """
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        # 辅助解码器：generator forward TBD:不要加这块的loss计算，只前向传播
        gen_outputs = self._generator_head_forward_train(x, data_samples)

        return seg_logits, gen_outputs

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    @property
    def with_auxiliary_neck(self) -> bool:
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,
                       'auxiliary_neck') and self.auxiliary_neck is not None

    def get_main_loss(self, inputs: Tensor, data_samples: SampleList):
        """Calculate losses from a batch of inputs and data samples.
        equal to loss() in encoder-decoder class
        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)

        # generator forward
        losses = dict()
        # 将两个头的loss计算进一步封装,这部分是主解码器
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        # if self.with_auxiliary_neck:

        # 辅助解码器：generator forward TBD:不要加这块的loss计算，只前向传播
        gen_outputs = self._generator_head_forward_train(x, data_samples)

        return losses, gen_outputs


    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits, gen_outs = self.inference(inputs, batch_img_metas, data_samples)

        return self.postprocess_result(seg_logits, data_samples, gen_outs)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        sig = inspect.signature(self.decode_head.forward)
        if 'batch_data_samples' in sig.parameters:
            return self.decode_head.forward(x, batch_data_samples=data_samples)

        else:
            return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg_custom/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict], data_samples) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg_custom/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits, gen_outs = self.encode_decode(inputs, batch_img_metas, data_samples)

        return seg_logits, gen_outs

    def inference(self, inputs: Tensor, batch_img_metas: List[dict], data_samples) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg_custom/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = batch_img_metas[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in batch_img_metas)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit, gen_outs = self.whole_inference(inputs, batch_img_metas, data_samples)

        return seg_logit, gen_outs

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    # 规定了一次迭代中模型训练流程行为
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper=None) -> Dict[str, torch.Tensor]:

        data = self.data_preprocessor(data, True)
        # discriminators, no updates to generator parameters.
        disc_optimizer_wrappers = optim_wrapper['discriminators']
        # main encoder-decoder 此时优化器早已被构建好，这里只是调用
        main_optimizer_wrapper = optim_wrapper['main_head']
        log_vars = dict()
        with main_optimizer_wrapper.optim_context(self):
            # forward main encoder-decoder and generators
            # generator, no updates to discriminator parameters.
            set_requires_grad(self.discriminators, False)
            # 这一步需要完成decode_head部分loss计算以及生成器输出计算
            results = self._main_run_forward(data, mode='loss')  # type: ignore
            if isinstance(results, tuple) and len(results) == 2:
                # 如果返回了两个值，那么第一个是losses_main，第二个是gen_outputs:dict
                losses_main, generator_outputs = results
                # forward discriminators with main net outputs for generators loss
                losses_gen = self._get_gen_loss(generator_outputs, data)
                losses_main.update(losses_gen)
                parsed_losses_main, log_vars_main = self.parse_losses(losses_main)  # type: ignore
                log_vars.update(log_vars_main)
                main_optimizer_wrapper.update_params(parsed_losses_main)
                # forward discriminators with main net outputs for discriminators loss
                set_requires_grad(self.discriminators, True)
                # optimize
                losses_disc, log_vars_disc = self._get_disc_loss(generator_outputs, data)
                disc_optimizer_wrappers.update_params(losses_disc)
                if 'loss' in log_vars_disc:
                    log_vars_disc['disc_loss'] = log_vars_disc.pop('loss')
                log_vars.update(log_vars_disc)
            else:
                # 如果只返回了一个值，那么这个值就是losses_main
                # for DEBUG
                print(f'WARNING: self._main_run_forward() only return one output！')
                losses_main = results
                parsed_losses_main, log_vars_main = self.parse_losses(losses_main)  # type: ignore
                log_vars.update(log_vars_main)
                main_optimizer_wrapper.update_params(parsed_losses_main)

        return log_vars

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._main_run_forward(data, mode='predict')  # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._main_run_forward(data, mode='predict')  # type: ignore

    def _main_run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def forward(self,
                inputs: Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`SegDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape (N, C, ...) in
                general.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.get_main_loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def get_module(self, module):
        """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel`
        interface.

        Args:
            module (MMDistributedDataParallel | nn.ModuleDict): The input
                module that needs processing.

        Returns:
            nn.ModuleDict: The ModuleDict of multiple networks.
        """
        if is_model_wrapper(module):
            return module.module

        return module

    def _get_gen_loss(self, generators_outputs, image_data):
        """Get the loss of generator.

        Args:
            outputs (dict): A dict of output.
            image_data (dict): A dict of RGB-X images.

        Returns:
            Tuple: Loss and a dict of log of loss terms.
        """
        target_domain_R = self._default_domain_R # X
        target_domain_X = self._default_domain_X # RGB
        losses = dict()
        # 是一个moduledict，其中包含了两个模态的判别器，存放为字典形式
        discriminators = self.get_module(self.discriminators)
        real_RGB, real_X = torch.split(image_data['inputs'], (3, 3), dim=1)
        # GAN loss for the rgb branch generator D_X
        fake_ab_X = torch.cat((real_RGB,
                             generators_outputs[f'gen_out.fake_{target_domain_R}']), 1)
        # discriminator_X forward
        fake_pred_X = discriminators[target_domain_R](fake_ab_X)
        losses['loss_gen_X'] = F.binary_cross_entropy_with_logits(
            fake_pred_X, 1. * torch.ones_like(fake_pred_X)) * self.gen_x_gan_loss_weight
        losses['loss_pixel_gen_X'] = F.l1_loss(
            real_X,
            generators_outputs[f'gen_out.fake_{target_domain_R}'],
            reduce='mean') * self.gen_x_pixel_loss_weight
        # GAN loss for the X branch generator D_RGB
        fake_ab_R = torch.cat((real_X,
                               generators_outputs[f'gen_out.fake_{target_domain_X}']), 1)
        # discriminator_R forward
        fake_pred_R = discriminators[target_domain_X](fake_ab_R)
        losses['loss_gen_R'] = F.binary_cross_entropy_with_logits(
            fake_pred_R, 1. * torch.ones_like(fake_pred_R)) * self.gen_rgb_gan_loss_weight
        losses['loss_pixel_gen_R'] = F.l1_loss(
            real_RGB,
            generators_outputs[f'gen_out.fake_{target_domain_X}'],
            reduce='mean') * self.gen_rgb_pixel_loss_weight
        return add_prefix(losses, 'gan')

    def _get_disc_loss(self, generators_outputs, image_data):
        """Get the loss of discriminator.

        Args:
            outputs (dict): A dict of output.
            image_data (dict): A dict of RGB-X images.

        Returns:
            Tuple: Loss and a dict of log of loss terms.
        """
        # GAN loss for the discriminator
        losses = dict()

        discriminators = self.get_module(self.discriminators)
        real_RGB, real_X = torch.split(image_data['inputs'], (3, 3), dim=1)
        target_domain_R = self._default_domain_R
        target_domain_X = self._default_domain_X
        # GAN loss for the disciriminator_X _R represents RGB branch
        fake_ab_X = torch.cat((real_RGB,
                             generators_outputs[f'gen_out.fake_{target_domain_R}']), 1)
        fake_pred_X = discriminators[target_domain_R](fake_ab_X.detach())
        losses['loss_disc_X_fake'] = F.binary_cross_entropy_with_logits(
            fake_pred_X, 0. * torch.ones_like(fake_pred_X))
        real_ab_X = torch.cat((real_RGB,
                             real_X), 1)
        real_pred_X = discriminators[target_domain_R](real_ab_X)
        losses['loss_disc_X_real'] = F.binary_cross_entropy_with_logits(
            real_pred_X, 1. * torch.ones_like(real_pred_X))
        # GAN loss for the disciriminator_RGB
        fake_ab_R = torch.cat((real_X,
                               generators_outputs[f'gen_out.fake_{target_domain_X}']), 1)
        fake_pred_R = discriminators[target_domain_X](fake_ab_R.detach())
        losses['loss_disc_R_fake'] = F.binary_cross_entropy_with_logits(
            fake_pred_R, 0. * torch.ones_like(fake_pred_R))
        real_ab_R = torch.cat((real_X,
                               real_RGB), 1)
        real_pred_R = discriminators[target_domain_X](real_ab_R)
        losses['loss_disc_R_real'] = F.binary_cross_entropy_with_logits(
            real_pred_R, 1. * torch.ones_like(real_pred_R))

        loss_d, log_vars_d = self.parse_losses(add_prefix(losses, 'gan'))
        loss_d *= 0.5

        return loss_d, log_vars_d

    def postprocess_result(self,
                           seg_logits: Tensor,
                           data_samples: OptSampleList = None,
                           gen_outs = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape
        target_domain_R = self._default_domain_R
        target_domain_X = self._default_domain_X

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)

            fake_X = gen_outs[f'gen_out.fake_{target_domain_R}'].squeeze(0)
            fake_RGB = gen_outs[f'gen_out.fake_{target_domain_X}'].squeeze(0)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred}),
                'gen_fake_X':
                PixelData(**{'data': fake_X}),
                'gen_fake_RGB':
                    PixelData(**{'data': fake_RGB})
            })

        return data_samples