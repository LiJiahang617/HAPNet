from mmseg_custom.datasets import CarlaDataset

data_root = '/media/ljh/data/carla_test'
data_prefix=dict(img_path='img_dir/training', seg_map_path='ann_dir/training')
# metainfo 中只保留以下 classes
dataset = CarlaDataset(data_root=data_root, data_prefix=data_prefix)

print(dataset.metainfo)

