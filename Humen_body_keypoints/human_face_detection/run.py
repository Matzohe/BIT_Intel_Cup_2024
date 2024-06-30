# 对于人脸识别的情况，我们将所有人脸的信息保存在Config字典里，并保存在本地
# 这一步的主要作用是进行初始化词典，同时能够增加新的人脸
# 要求dataset的情况为，一个文件夹下有所有图片
from human_face_setup import config_default, setup_database
import torch
from human_face_dataset import HumanFaceDataset


def run(data_path, new_config=True, config_path=None):
    config = config_default()
    if not new_config:
        if config_path is None:
            raise ValueError("导入config文件时，其路径不能为None")
        config = torch.load(config_path)
    dataset = HumanFaceDataset(data_path)
    img_list = []
    for i in range(len(dataset)):
        img_list.append(dataset[i])
    config = setup_database(img_list=img_list, config_=config)
    return config
