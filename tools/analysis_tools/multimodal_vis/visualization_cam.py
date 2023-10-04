"""Use the pytorch-grad-cam tool to visualize Class Activation Maps (CAM).
requirement: pip install grad-cam
"""
import os
from argparse import ArgumentParser
from typing import Type, Union, Sequence
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine_custom.model import revert_sync_batchnorm
from mmengine_custom.dataset import Compose
from PIL import Image
from pytorch_grad_cam import GradCAM, LayerCAM, XGradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

from mmengine_custom import Config
from mmseg_custom.apis import init_model, show_result_pyplot
from mmseg_custom.structures import SegDataSample
from mmseg_custom.models import BaseSegmentor
from mmseg_custom.utils import register_all_modules, SampleList


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


def _preprare_mulitmodal_data(imgs, anos, model):

    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') in ['LoadAnnotations', 'LoadCityscapesAnnotations']:
            cfg.test_pipeline.remove(t)

    is_batch = True
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
        anos = [anos]
        is_batch = False

    if isinstance(imgs[0], np.ndarray):
        cfg.test_pipeline[0]['type'] = 'LoadImageFromNDArray'

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for (img, ano) in zip(imgs, anos):
        if isinstance(img, np.ndarray) and isinstance(ano, np.ndarray):
            data_ = dict(img=img)
            data_['ano'] = ano
        else:
            data_ = dict(img_path=img)
            data_['ano_path'] = ano
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

    return data, is_batch

ImageType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]

def inference_multimodel(img: ImageType, ano: ImageType,
                         model: BaseSegmentor) -> Union[SegDataSample, SampleList]:
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        :obj:`SegDataSample` or list[:obj:`SegDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the segmentation results directly.
    """
    # prepare data
    data, is_batch = _preprare_mulitmodal_data(img, ano, model)

    # forward the model
    with torch.no_grad():
        results = model.test_step(data)

    return results if is_batch else results[0]


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('ano', help='Another file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    """
    to handle sometimes original image is too large to parse, may cause Cuda memory error.
    resize func is implemented using cv2 backends, which is (W, H)
    when large is used and report cuda out of memory, modify loss.backward(retain_graph=False) in base_cam.py
    may solve the issue, but it cost users more time to run specific class heatmap visualization, because once 
    graph is deleted, you can only get one class map.
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
    result = inference_multimodel(args.img, args.ano, model)
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
    ano_img = np.array(Image.open(args.ano).convert('RGB'))
    ano_copy = np.copy(ano_img)
    ano_resized = cv2.resize(ano_copy, size)
    concat_img = np.concatenate((image_resized, ano_resized), axis=2)
    height, width = concat_img.shape[0], concat_img.shape[1]
    rgb_img = np.float32(image_resized) / 255
    concat_img = np.float32(concat_img) / 255
    config = Config.fromfile(args.config)
    image_mean = config.data_preprocessor['mean']
    image_std = config.data_preprocessor['std']
    input_tensor = preprocess_image(
        concat_img,
        mean=[x / 255 for x in image_mean],
        std=[x / 255 for x in image_std])

    # Grad CAM(Class Activation Maps)
    # Can also be LayerCAM, XGradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM
    targets = [
        SemanticSegmentationTarget(category, mask_float,
                                   (height, width))
    ]
    # model will do forward once and backward once here
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