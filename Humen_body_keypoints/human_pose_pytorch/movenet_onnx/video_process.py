import cv2
from dataloader import cv2_video_process
from draw import cv2_draw
import torch
import os
import numpy as np
import torch.onnx
import onnx
import onnxruntime
import time


def video_process(video_path, sess):
    key_point_save_list = []
    capture = cv2.VideoCapture(video_path)
    if capture.isOpened() is False:
        raise RuntimeError("打开视频文件失败")
    while capture.isOpened():
        ret, frame = capture.read()
        if ret is True:
            image, display_image = cv2_video_process(frame)
            output = sess.run(None, {'input': image})[0]
            key_point_save_list.append(output)
            # b, g, r = cv2.split(display_image)
            # display_image = cv2.merge((r, g, b))
            # output_overlay = cv2_draw(display_image, output)
            # cv2.imshow('video', output_overlay)
        else:
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    capture.release()
    return key_point_save_list


def key_point_save_function(data_root_dir_, sess_, save_root):
    pose_root = os.listdir(data_root_dir_)
    info_dict = {}
    for i, each in enumerate(pose_root):
        pose_list = []
        video_root = os.path.join(data_root_dir_, each)
        video_dir = os.listdir(video_root)
        for rt in video_dir:
            video_path = os.path.join(video_root, rt)
            key_list = video_process(video_path, sess_)
            pose_list = pose_list + key_list
        pose_list = np.concatenate(pose_list)
        info_dict[i] = torch.from_numpy(pose_list)
    torch.save(info_dict, save_root)


if __name__ == "__main__":
    onnx_model = r"G:\Module_Parameter\humen_pose\movenet\single-pose-thunder\model.onnx"
    sess = onnxruntime.InferenceSession(onnx_model)
    train_data_root_dir = r"M:\data package\human_pose_detection\video"
    test_data_root_dir = r"M:\data package\human_pose_detection\video_val"
    train_data_save_root = r"M:\data package\human_pose_detection\train_keypoint_dict.pt"
    test_data_save_root = r"M:\data package\human_pose_detection\test_keypoint_dict.pt"
    key_point_save_function(train_data_root_dir, sess, train_data_save_root)
    key_point_save_function(test_data_root_dir, sess, test_data_save_root)

