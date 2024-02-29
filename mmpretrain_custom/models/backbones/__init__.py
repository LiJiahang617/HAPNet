# Copyright (c) OpenMMLab. All rights reserved.
from .twin_convnext import TwinConvNeXt
from .convnext import ConvNeXt
from .share_convnext import ShareConvNeXt
from .sumtwin_convnext import SumTwinConvNeXt
from .convnext_adapter import ConvNeXtAdapter
from .convnext_concat_adapter import ConvNeXtCatAdapter
from .beit_adapter import BEiTAdapter
from .sharesum_convnext import ShareSumConvNeXt
from .sharecat_convnext import ShareCatConvNeXt
from .beit_rgbx_cat_adapter import BEiTAdapter_rgbxcat
from .beit_rgbx_sum_adapter import BEiTAdapter_rgbxsum
from .beit_adapter_ori import BEiTAdapter_ori
from .beit_ori import BEiT_ori
from .beit_adapter_mmlab import BEiTAdapter_mmlab
from .spm_rgbx_sum_adapter import BEiTAdapter_spmsum
from .beit_sum_fapn_adapter import BEiTAdapter_rgbxsum_fapn
from .beit_sum_fapn_c1_c2 import BEiTAdapter_rgbxsum_fapn_c1_c2_relu
from .beit_sum_fapn_c1_c2_worelu import BEiTAdapter_rgbxsum_fapn_c1_c2_worelu
# for ablation study of ECCV 2024
from .beit_ablation import BEiT_ablation
from .beit_sum_adapter_separate_patch import BEiTAdapter_sep_patch
from .beit_sum_adapter_share_patch import BEiTAdapter_share_patch
from .beit_sum_adapter_thermal_patch_alone import BEiTAdapter_thermal_patch_alone
from .beit_sum_adapter_rgbt_conv_patch import BEiTAdapter_rgbt_conv_patch
from .beit_sum_adapter_concat_share_patch import BEiTAdapter_concat_share_patch
from .beit_ablation_cat_share import BEiT_cat_share
from .beit_sum_adapter_sum_share_patch import BEiTAdapter_sum_share_patch
from .double_convnext_adapter import DoubleConvNeXtAdapter
from .beit_rgb_alone_mpm_adapter import BEiTAdapter_patch_rgb_alone_mpm_rgb_alone
from .beit_thermal_alone_mpm_adapter import BEiTAdapter_patch_rgb_alone_mpm_thermal_alone
from .beit_thermal_alone_patch_rgb_alone_mpm_adapter import BEiTAdapter_patch_thermal_alone_mpm_rgb_alone
from .beit_thermal_alone_patch_thermal_alone_mpm_adapter import BEiTAdapter_patch_thermal_alone_mpm_thermal_alone
from .beit_rgb_thermal_patch_rgb_alone_mpm_adapter import BEiTAdapter_patch_rgb_thermal_mpm_rgb_alone
from .beit_rgb_thermal_patch_rgb_thermal_mpm_adapter import BEiTAdapter_patch_rgb_thermal_mpm_rgb_thermal
from .beit_rgb_thermal_patch_thermal_alone_mpm_adapter import BEiTAdapter_patch_rgb_thermal_mpm_thermal_alone
from .dual_beit import DualBEiT
from .twin_beit_sum import TwinBeiTSum
from .twin_beit_concat import TwinBeiTCat

__all__ = [
    'ConvNeXt', 'ShareConvNeXt', 'ConvNeXtCatAdapter',
    'BEiTAdapter', 'ShareSumConvNeXt', 'ShareCatConvNeXt',
    'TwinConvNeXt', 'SumTwinConvNeXt', 'ConvNeXtAdapter', 'BEiT_ori',
    'BEiTAdapter_rgbxcat', 'BEiTAdapter_rgbxsum', 'BEiTAdapter_ori',
    'BEiTAdapter_mmlab', 'BEiTAdapter_spmsum', 'BEiTAdapter_rgbxsum_fapn',
    'BEiTAdapter_rgbxsum_fapn_c1_c2_relu', 'BEiTAdapter_rgbxsum_fapn_c1_c2_worelu',
    # for ablation study of ECCV 2024
    'BEiT_ablation', 'BEiTAdapter_sep_patch', 'BEiTAdapter_share_patch',
    'BEiTAdapter_thermal_patch_alone', 'BEiTAdapter_rgbt_conv_patch',
    'BEiTAdapter_concat_share_patch', 'BEiT_cat_share', 'BEiTAdapter_sum_share_patch',
    'DoubleConvNeXtAdapter', 'BEiTAdapter_patch_rgb_alone_mpm_rgb_alone', 'BEiTAdapter_patch_rgb_alone_mpm_thermal_alone',
    'BEiTAdapter_patch_thermal_alone_mpm_rgb_alone', 'BEiTAdapter_patch_thermal_alone_mpm_thermal_alone',
    'BEiTAdapter_patch_rgb_thermal_mpm_rgb_alone', 'BEiTAdapter_patch_rgb_thermal_mpm_rgb_thermal',
    'BEiTAdapter_patch_rgb_thermal_mpm_thermal_alone', 'DualBEiT', 'TwinBeiTSum', 'TwinBeiTCat'

]
