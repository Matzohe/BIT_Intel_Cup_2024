import cv2
import os
from torch.utils.data import Dataset


class HumanFaceDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.path_dir = os.listdir(data_path)
        self.path_list = [os.path.join(data_path, self.path_dir[i]) for i in range(len(self.path_dir))]

    def __getitem__(self, item):
        img = cv2.imread(self.path_list[item])
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])
        return img

    def __len__(self):
        return len(self.path_list)
