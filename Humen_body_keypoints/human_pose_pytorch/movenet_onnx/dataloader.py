from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import cv2
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
import time


def cv2_image_process(image_path):
    time1 = time.time()
    new_image = cv2.imread(image_path)
    b, g, r = cv2.split(new_image)
    new_image = cv2.merge((r, g, b))
    image = cv2.resize(new_image, (256, 256))
    image = np.expand_dims(image, axis=0).astype(np.int32)
    display_image = new_image
    # display_image = cv2.resize(new_image, (1280, 1280), interpolation=cv2.INTER_LINEAR)
    # display_image = np.expand_dims(display_image, axis=0)
    time2 = time.time()
    print("data process time={}".format(time2 - time1))
    return image, display_image


def cv2_video_process(image_input):
    new_image = image_input
    b, g, r = cv2.split(new_image)
    new_image = cv2.merge((r, g, b))
    image = cv2.resize(new_image, (256, 256))
    image = np.expand_dims(image, axis=0).astype(np.int32)
    display_image = new_image

    # display_image = cv2.resize(new_image, (1280, 1280), interpolation=cv2.INTER_LINEAR)
    # display_image = np.expand_dims(display_image, axis=0)
    return image, display_image


def image_process(image_path):
    loading_start = time.time()

    new_image = Image.open(image_path)
    image = new_image
    open_time = time.time()
    image = torch.from_numpy(np.array(image)) * 255
    if image_path.endswith(".png"):
        image = image[:3, :, :]
    # transform = transforms.Compose(
    #     [transforms.ToTensor()])
    # if image_path.endswith(".png"):
    #     image = (transform(image) * 255)[: 3, :, :]
    # else:
    #     image = transform(image) * 255
    to_tensor_transform_time = time.time()
    image = image.unsqueeze(dim=0)
    input_size = 256
    image = F.interpolate(image, size=input_size, mode='bilinear', align_corners=False)
    image = image.permute(0, 2, 3, 1).to(torch.int32).numpy()
    transform_time = time.time()

    # display_image = Image.open(image_path)
    # display_image = transform(display_image) * 255
    display_image = new_image
    display_image = torch.from_numpy(np.array(display_image))
    display_image = display_image.unsqueeze(dim=0)
    display_image = F.interpolate(display_image, size=1280, mode='bilinear')
    display_image = display_image.permute(0, 2, 3, 1)
    display_image = display_image.to(torch.int32).numpy()

    print("data load time={}".format(open_time - loading_start))
    print("fig to tensor time={}".format(to_tensor_transform_time - open_time))
    print("fig process time={}".format(transform_time - open_time))

    return image, display_image


class HumanPoseDataset(Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.label = data_root[-1]
        self.root_dir = [os.path.join(data_root, each) for each in os.listdir(data_root)]

    def __getitem__(self, item):
        image_path = self.root_dir[item]
        image, display_image = image_process(image_path)
        return image, display_image

    def __len__(self):
        return len(self.root_dir)


if __name__ == '__main__':
    data_path = r"M:\{8AFF922C-29FA-E5DD-E52D-4A771FC41574}.jpg"
    cv2_image_process(data_path)