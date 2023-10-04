python tools/analysis_tools/multimodal_vis/visualization_cam.py \
/media/ljh/data/Cityscapes/images/test/munich_000333_000019_leftImg8bit.png \
/media/ljh/data/Cityscapes/sne/test/munich_000333_000019_normal.png \
configs/mit_mlp/mm0-255_mit-b0_allmlp_cityscapes-512x1024_norm.py \
work_dirs/mm0-255_mit-b0_allmlp_cityscapes-512x1024_norm/best_mIoU_iter_85000.pth \
--width 2048 --height 1024
