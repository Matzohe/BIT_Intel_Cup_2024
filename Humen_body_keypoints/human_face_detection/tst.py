import torch
import cv2
from svd_seperate import compute_similarity


def function_test(data_path, config):
    # There is a function to test someone not in the database
    img = cv2.imread(data_path)
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    simi_info = []
    for key, value in config["face_info_dict"].items():
        _, distance = compute_similarity(img, value)
        simi_info.append(distance)
    ts = torch.tensor(simi_info)
    idx = torch.argmax(ts, dim=-1)
    person_name = "S_" + str(idx)
    trust_value = ts[idx]
    return person_name, trust_value
