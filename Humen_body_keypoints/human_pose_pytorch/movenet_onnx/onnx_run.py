from draw import draw_prediction_on_image
from gradio_generate import gradio_generate
import torch
import torch.onnx
import onnxruntime
import numpy as np
from dataloader import cv2_image_process
from video_process import video_process
from PIL import Image
from dataloader import image_process
import time
from draw import cv2_draw


def running_function(datapath): # 这个是img版本的输入，下面搞一个处理视频输入的方程
    start_time = time.time()
    image, display_image = cv2_image_process(datapath)
    data_process_time = time.time()
    output = sess.run(None, {'input': image})
    neuro_network_process = time.time()
    output = torch.tensor(output[0])
    output_class = classification.run(None, {"input": output.numpy()})


    # output_overlay = draw_prediction_on_image(
    #     np.squeeze(display_image, axis=0), output)
    output_overlay = cv2_draw(display_image, output, output_class[0])

    output_overlay = Image.fromarray(output_overlay)
    overlay_time = time.time()
    print("Data process time={}".format(data_process_time - start_time))
    print("Neuro network  process time={}".format(neuro_network_process - data_process_time))
    print("output overlay time={}".format(overlay_time - neuro_network_process))

    return output_overlay


if __name__ == "__main__":
    onnx_model = r"G:\Module_Parameter\humen_pose\movenet\single-pose-thunder\model.onnx"
    classification_onnx_model = r"G:\python_program\Humen_body_keypoints\human_pose_pytorch\classification.onnx"
    sess = onnxruntime.InferenceSession(onnx_model)
    classification = onnxruntime.InferenceSession(classification_onnx_model)
    gradio_generate(running_function)
    

