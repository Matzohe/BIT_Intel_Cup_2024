import torch
from svd_seperate import compute_singular_value
from Config import ConfigSet


def config_default():
    config = ConfigSet(face_info_dict={},
                       number=0,)
    return config


def setup_database(img_list, config_, auto_setup=True, human_id=None, loader=False, config_path=None):
    face_info_dict = config_['face_info_dict']
    if loader:
        if config_path is None:
            raise ValueError("在从本地导入config数据时，config_path不能为None")
        config_ = torch.load(config_path)
    if auto_setup:
        idx = config_["number"]
        human_id = "S_" + str(idx)
        config_.add_item({"number": idx + 1})
    else:
        if human_id is None:
            raise ValueError("如果自定义姓名，human_id值不能为None")
        if not isinstance(human_id, str):
            human_id = str(human_id)

    if face_info_dict.get(human_id, -1) != -1:
        print("此human id已经被占用")
    else:
        avg_img = 0
        for each in img_list:
            avg_img += each / len(img_list)
        singular_value = compute_singular_value(avg_img)
        face_info_dict[human_id] = singular_value
        config_.add_item({"face_info_dict": face_info_dict})
    return config_
