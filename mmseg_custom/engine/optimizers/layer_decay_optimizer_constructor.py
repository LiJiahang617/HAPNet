# Copyright (c) OpenMMLab. All rights reserved.
import json
import warnings

from mmengine_custom.dist import get_dist_info
from mmengine_custom.logging import print_log
from mmengine_custom.optim import DefaultOptimWrapperConstructor

from mmseg_custom.registry import OPTIM_WRAPPER_CONSTRUCTORS


def get_layer_id_for_convnext(var_name, max_layer_id):
    """Get the layer id to set the different learning rates in ``layer_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_layer_id (int): Maximum number of backbone layers.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    else:
        return max_layer_id + 1


def get_stage_id_for_convnext(var_name, max_stage_id):
    """Get the stage id to set the different learning rates in ``stage_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_stage_id (int): Maximum number of backbone layers.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        return 0
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return max_stage_id - 1


def get_layer_id_for_vit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('decode_head.mask_embed'):
        return 0
    elif var_name.startswith('decode_head.cls_embed'):
        return 0
    elif var_name.startswith('decode_head.level_embed'):
        return 0
    elif var_name.startswith('decode_head.query_embed'):
        return 0
    elif var_name.startswith('decode_head.query_feat'):
        return 0
    elif var_name.startswith('backbone.x_modality_encoder.downsample_layers'):
        return 0
    elif var_name.startswith('backbone.x_modality_encoder.stages'):
        stage_id = int(var_name.split('.')[3])
        return stage_id + 1
    elif (var_name.startswith('backbone.blocks') or
          var_name.startswith('backbone.layers')):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return max_layer_id - 1


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for ConvNeXt,
    BEiT and MAE.
    """

    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """

        parameter_groups = {}
        print_log(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        print_log('Build LearningRateDecayOptimizerConstructor  '
                  f'{decay_type} {decay_rate} - {num_layers}')
        weight_decay = self.base_wd
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token'):
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            if 'layer_wise' in decay_type:
                if 'ConvNeXt' in module.backbone.__class__.__name__:
                    layer_id = get_layer_id_for_convnext(
                        name, self.paramwise_cfg.get('num_layers'))
                    print_log(f'set param {name} as id {layer_id}')
                elif 'BEiT' in module.backbone.__class__.__name__ or \
                     'MAE' in module.backbone.__class__.__name__:
                    layer_id = get_layer_id_for_vit(name, num_layers)
                    print_log(f'set param {name} as id {layer_id}')
                else:
                    raise NotImplementedError()
            elif decay_type == 'stage_wise':
                if 'ConvNeXt' in module.backbone.__class__.__name__:
                    layer_id = get_stage_id_for_convnext(name, num_layers)
                    print_log(f'set param {name} as id {layer_id}')
                else:
                    raise NotImplementedError()
            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            print_log(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class LayerDecayOptimizerConstructor(DefaultOptimWrapperConstructor):
    # for now, this class is just for BEiT-Adapter: ViT and ConvNeXt only.
    def add_params(self, params, module, prefix='', is_dcn_module=None):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.
        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        parameter_groups = {}
        print(self.paramwise_cfg)
        # for ViT
        vit_num_layers = self.paramwise_cfg.get('vit_num_layers') + 2
        # drawbacks: ViT and x_modality encoder share the decay rate, can implement dependent weights
        # TODO: x_encoder_num_layers now are not useful, can only use vit num in func below
        decay_rate = self.paramwise_cfg.get('decay_rate')
        # for x_modality_encoder
        if self.paramwise_cfg.get('x_encoder_num_layers'):
            x_encoder_num_layers = self.paramwise_cfg.get('x_encoder_num_layers') + 2
            print_log('Build LearningRateDecayOptimizerConstructor  '
                      f'ViT {decay_rate} - {vit_num_layers}'
                      f'x_modality_encoder {decay_rate} - {x_encoder_num_layers}')
        else:
            print_log('Build LearningRateDecayOptimizerConstructor  '
                      f'ViT {decay_rate} - {vit_num_layers}')
        # print('Build LayerDecayOptimizerConstructor %f - %d' %
        #       (layer_decay_rate, num_layers))
        weight_decay = self.base_wd

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') \
                    or name in ('pos_embed', 'cls_token', 'visual_embed'):
                # or "relative_position_bias_table" in name:
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay

            layer_id = get_layer_id_for_vit(name, vit_num_layers)
            group_name = 'layer_%d_%s' % (layer_id, group_name)

            if group_name not in parameter_groups:
                scale = decay_rate**(vit_num_layers - layer_id - 1)

                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            # print('Param groups = %s' % json.dumps(to_display, indent=2))

        # state_dict = module.state_dict()
        # for group_name in parameter_groups:
        #     group = parameter_groups[group_name]
        #     for name in group["param_names"]:
        #         group["params"].append(state_dict[name])
        params.extend(parameter_groups.values())
