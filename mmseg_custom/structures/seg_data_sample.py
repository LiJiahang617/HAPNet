# Copyright (c) OpenMMLab. All rights reserved.
from mmengine_custom.structures import BaseDataElement, PixelData


class SegDataSample(BaseDataElement):
    """A data structure interface of MMSegmentation. They are used as
    interfaces between different components.

    The attributes in ``SegDataSample`` are divided into several parts:

        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
        - ``seg_logits``(PixelData): Predicted logits of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine_custom.structures import PixelData
         >>> from mmseg_custom.structures import SegDataSample

         >>> data_sample = SegDataSample()
         >>> img_meta = dict(img_shape=(4, 4, 3),
         ...                 pad_shape=(4, 4, 3))
         >>> gt_segmentations = PixelData(metainfo=img_meta)
         >>> gt_segmentations.data = torch.randint(0, 2, (1, 4, 4))
         >>> data_sample.gt_sem_seg = gt_segmentations
         >>> assert 'img_shape' in data_sample.gt_sem_seg.metainfo_keys()
         >>> data_sample.gt_sem_seg.shape
         (4, 4)
         >>> print(data_sample)
        <SegDataSample(

            META INFORMATION

            DATA FIELDS
            gt_sem_seg: <PixelData(

                    META INFORMATION
                    img_shape: (4, 4, 3)
                    pad_shape: (4, 4, 3)

                    DATA FIELDS
                    data: tensor([[[1, 1, 1, 0],
                                 [1, 0, 1, 1],
                                 [1, 1, 1, 1],
                                 [0, 1, 0, 1]]])
                ) at 0x1c2b4156460>
        ) at 0x1c2aae44d60>

        >>> data_sample = SegDataSample()
        >>> gt_sem_seg_data = dict(sem_seg=torch.rand(1, 4, 4))
        >>> gt_sem_seg = PixelData(**gt_sem_seg_data)
        >>> data_sample.gt_sem_seg = gt_sem_seg
        >>> assert 'gt_sem_seg' in data_sample
        >>> assert 'sem_seg' in data_sample.gt_sem_seg
    """

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self) -> None:
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData) -> None:
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self) -> None:
        del self._pred_sem_seg

    @property
    def seg_logits(self) -> PixelData:
        return self._seg_logits

    @seg_logits.setter
    def seg_logits(self, value: PixelData) -> None:
        self.set_field(value, '_seg_logits', dtype=PixelData)

    @seg_logits.deleter
    def seg_logits(self) -> None:
        del self._seg_logits

    @property
    def gen_fake_X(self) -> PixelData:
        return self._gen_fake_X

    @gen_fake_X.setter
    def gen_fake_X(self, value: PixelData) -> None:
        self.set_field(value, '_gen_fake_X', dtype=PixelData)

    @gen_fake_X.deleter
    def gen_fake_X(self) -> None:
        del self._gen_fake_X

    @property
    def gen_fake_RGB(self) -> PixelData:
        return self._gen_fake_RGB

    @gen_fake_RGB.setter
    def gen_fake_RGB(self, value: PixelData) -> None:
        self.set_field(value, '_gen_fake_RGB', dtype=PixelData)

    @gen_fake_RGB.deleter
    def gen_fake_RGB(self) -> None:
        del self._gen_fake_RGB

    @property
    def ta_mask_X(self) -> PixelData:
        return self._ta_mask_X

    @ta_mask_X.setter
    def ta_mask_X(self, value: PixelData) -> None:
        self.set_field(value, '_ta_mask_X', dtype=PixelData)

    @ta_mask_X.deleter
    def ta_mask_X(self) -> None:
        del self._ta_mask_X