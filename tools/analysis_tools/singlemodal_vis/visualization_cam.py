# Copyright (c) OpenMMLab. All rights reserved.
"""Use the pytorch-grad-cam tool to visualize Class Activation Maps (CAM).
requirement: pip install grad-cam
"""
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine_custom.model import revert_sync_batchnorm
from PIL import Image
from pytorch_grad_cam import GradCAM, LayerCAM, XGradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

from mmengine_custom import Config
from mmseg_custom.apis import inference_model, init_model, show_result_pyplot
from mmseg_custom.utils import register_all_modules


class SemanticSegmentationTarget:
    """wrap the model.
    requirement: pip install grad-cam
    Args:
        category (int): Visualization class.
        mask (ndarray): Mask of class.
        size (tuple): Image size.
    """

    def __init__(self, category, mask, size):
        self.category = category
        self.mask = torch.from_numpy(mask)
        self.size = size
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        model_output = torch.unsqueeze(model_output, dim=0)
        model_output = F.interpolate(
            model_output, size=self.size, mode='bilinear')
        model_output = torch.squeeze(model_output, dim=0)

        return (model_output[self.category, :, :] * self.mask).sum()


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    """
    to handle sometimes original image is too large to parse, may cause Cuda memory error.
    resize func is implemented using cv2 backends, which is (W, H)
    users would better to set output size according to their model settings.
    """
    parser.add_argument('--width', type=int, help='Width of the output image')
    parser.add_argument('--height', type=int, help='Height of the output image')
    parser.add_argument(
        '--out-file',
        default='prediction.png',
        help='Path to output prediction file')
    parser.add_argument(
        '--cam-file',
        default='vis_cam.png',
        help='Path to output cam file')
    parser.add_argument(
        '--target-layers',
        default='decode_head.convs[3].activate',
        help='Target layers to visualize CAM')
    parser.add_argument(
        '--category-index',
        default='11',
        help='Category to visualize CAM')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    register_all_modules()
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    # list all module names
    for name, module in model.named_modules():
        print(name)
    # test a single image
    result = inference_model(model, args.img)

    # show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        draw_gt=False,
        show=False if args.out_file is not None else True,
        out_file=os.path.join(os.path.dirname(args.checkpoint), args.out_file))

    # result data conversion
    prediction_data = result.pred_sem_seg.data
    pre_np_data = prediction_data.cpu().numpy().squeeze(0)

    target_layers = args.target_layers
    target_layers = [eval(f'model.{target_layers}')]

    category = int(args.category_index)
    mask_float = np.float32(pre_np_data == category)
    # mask resize
    size = (args.width, args.height)
    mask_float = cv2.resize(mask_float, size)
    # data processing
    image = np.array(Image.open(args.img).convert('RGB'))
    image_copy = np.copy(image)
    image_resized = cv2.resize(image_copy, size)
    height, width = image_resized.shape[0], image_resized.shape[1]
    rgb_img = np.float32(image_resized) / 255
    config = Config.fromfile(args.config)
    image_mean = config.data_preprocessor['mean']
    image_std = config.data_preprocessor['std']
    input_tensor = preprocess_image(
        rgb_img,
        mean=[x / 255 for x in image_mean],
        std=[x / 255 for x in image_std])

    # Grad CAM(Class Activation Maps)
    # Can also be LayerCAM, XGradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM
    targets = [
        SemanticSegmentationTarget(category, mask_float,
                                   (height, width))
    ]
    with GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=torch.cuda.is_available()) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # save cam file
        Image.fromarray(cam_image).save(os.path.join(os.path.dirname(args.checkpoint), args.cam_file))


if __name__ == '__main__':
    main()