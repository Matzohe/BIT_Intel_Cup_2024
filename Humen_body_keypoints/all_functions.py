import cv2
import onnxruntime
import pyaudio
import torch
import dlib
import os
import gradio as gr
from human_face_detection.Config import ConfigSet
from utils import pose_detect_show, setup_database, face_test, get_human_face_point, get_tired_thread, get_fatigue_score, opencv_face_saver
import threading
from voice.chat import ivw_wakeup, speakout
import wave
import numpy as np
import time
from openvino.runtime import Core



def generate_gui(input_function):
    gui = gr.Interface(
        fn=input_function,
        inputs=None,
        outputs=[gr.Image(label="video"), gr.Textbox(label="Ciallo～(∠・ω< )⌒★")],
        live=True,
    )
    return gui


def get_openvino_model(parent_path):
    model_path = parent_path + ".xml"
    weights_path = parent_path + ".bin"
    ie = Core()
    model = ie.read_model(model=model_path, weights=weights_path)
    compile_model = ie.compile_model(model=model, device_name="CPU")
    return compile_model

def cv2_image_process(img):
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])


class MyFunction:
    # 将一些需要用到的变量或者需要保存的中间参数保存在这个类内部
    def __init__(self, config):
        face_config_path = config['face_config_path']
        self.config_ = config
        self.names = ["毛子鸣", "刘卓", "侯锦坤", "S_0", "S_1", "S_2"]
        if face_config_path is None:
            raise RuntimeError("config path 不能为空，请指定保存人脸信息的路径")
        if not os.path.exists(face_config_path):
            self.face_config = ConfigSet(face_info_dict={}, number=0,)
            torch.save(self.face_config, face_config_path)
        else:
            self.face_config = torch.load(face_config_path)
        self.face_detect_list = []  # 用于保存进行人脸识别时记录的图像
        self.user = self.names[0]  # 用户default量
        self.pose_list = ["Normal posture", "Over Sticking", "paralyzed", "legs crossed", "hunchbacked"]

        # 两个坐姿检测中使用到的模型
        self.sess = onnxruntime.InferenceSession(config['sess_path'])
        # self.sess = get_openvino_model(config['sess_path'].split(".")[0])

        # self.classification = onnxruntime.InferenceSession(config['classification_path'])
        self.classification = get_openvino_model(config['classification_path'].split(".")[0])

        # 人脸识别中使用到的模型,第一个模型是检测是否存在人脸的，第二个模型是将人脸的68个点位进行标注
        self.detector = dlib.get_frontal_face_detector()
        self.face_pointer = dlib.shape_predictor(config['pointer_path'])

        # 下面保存当网络没有在运行时在展示时展示的图像和文本, 以及保存一个网络的状态，看是否在运行
        self.default_image = cv2_image_process(cv2.imread(config['default_img_path']))
        self.default_text = "闲置中...O.O"
        self.is_rest = False
        self.function = 0

        # 下面的参数是人脸疲劳度使用的参数
        self.tired_score = 0
        self.ear_list = []
        self.mar_list = []
        self.is_first = True  # 这个参数的作用是用于标注是否刚开始检测，如果刚开始检测，则根据这个人的特点来确定ear和mar的阈值
        self.eye_thread = 0
        self.mouth_thread = 0
        self.beta = config['beta']
        self.info_path = config['path']

        # 信息保存的位置
        self.tired_root = config['tired_root']
        self.pose_root = config['pose_root']

        self.cv2_recognize_root = config['cv2_recognize_root']
        self.cv2_recognize_root_list = [os.path.join(self.cv2_recognize_root, each) for each in os.listdir(self.cv2_recognize_root)]
        self.cv2_save_root = config['cv2_root']
        self.all_number = 0

        self.user_index = 0

        self.personal_root_list = [os.path.join(config['personal_root'], each) for each in os.listdir(config['personal_root'])]

        self.pose_0_number = 0
        self.pose_1_number = 0
        self.pose_2_number = 0
        self.pose_3_number = 0
        self.pose_4_number = 0

        self.make_voice = False

        self.word_list = ["您的坐姿非常标准，保持正确的坐姿不仅能够展现你的优雅姿态，还能有效预防背部和颈部疼痛，坚持下去，你的身体会感激你现在的努力！",
                          "您似乎特别喜欢过度挺直您的背部，您可以轻轻地放松你的肩膀和背部，让脊柱保持自然的S形曲线，这样不仅能让您感觉更舒适，还能促进更健康的身体姿态。",
                          "您似乎特别喜欢瘫在座椅上，试着将你的背部轻轻靠向椅背，双脚平放地面，这不仅能减少身体的压力，还能让你看起来更加精神和自信。",
                          "您似乎特别喜欢翘二郎腿，轻轻放下二郎腿，让双脚平放地面，这样可以帮助改善血液循环，同时保持身体的平衡和舒适。",
                          "您的坐姿稍微有点驼背，轻轻挺直你的背部，想象有一根绳子从头顶向上拉，这不仅能让你的身姿更加挺拔，还能减轻背部的压力，预防驼背带来的不适。"]

    def get_faces(self, face_rectangles):
        rec_list = [[rec.left(), rec.top(), rec.right(), rec.bottom()] for rec in face_rectangles]
        return rec_list[0]

    def tired_function(self, frame):
        frame, ear, mar = get_human_face_point(frame, self.detector, self.face_pointer)
        if ear == -1 and mar == -1:
            return frame, "现在您的疲劳度为{}".format(self.tired_score)
        else:
            self.ear_list.append(ear)
            self.mar_list.append(mar)
        if len(self.ear_list) % 30 == 0 and len(self.ear_list) > 0:
            if self.is_first:
                self.eye_thread, self.mouth_thread = get_tired_thread(self.ear_list[-30:],
                                                                      self.mar_list[-30:])
                self.is_first = False
            else:
                self.tired_score = \
                    get_fatigue_score(self.eye_thread, self.mouth_thread,
                                      self.ear_list[-30:], self.mar_list[-30:]) \
                    + self.beta * self.tired_score
                with open(self.tired_root, 'w', encoding='utf-8') as f:
                    nb = int(self.tired_score)
                    f.write(str(nb))
                if self.tired_score >= 60:
                    data_config = torch.load(self.personal_root_list[self.user_index])
                    data_config['tired_number'] += 0.1
                    torch.save(data_config, self.personal_root_list[self.user_index])
                    speakout("您现在很疲劳了，请注意休息")
                if self.tired_score > 99:
                    self.tired_score = 99
        return frame, "现在您的疲劳度为{}".format(self.tired_score)

    def pose_position(self, frame):
        try:
            frame, _, output_class = pose_detect_show(frame, self.sess, self.classification, openvino_model=True)
            # eval("self.pose_{}_number += 1".format(output_class))
            if output_class == 0:
                self.pose_0_number += 1
            elif output_class == 1:
                self.pose_1_number += 1
            elif output_class == 1:
                self.pose_2_number += 1
            elif output_class == 3:
                self.pose_3_number += 1
            elif output_class == 4:
                self.pose_4_number += 1
            if self.all_number <= 30:
                self.all_number += 1
            else:
                self.all_number = 0
                with open(self.pose_root, 'w', encoding='utf-8') as f:
                    f.write(str(output_class))
                data_info = torch.load(self.personal_root_list[self.user_index])
                data_info["pose_0"] += self.pose_0_number / 30
                data_info["pose_1"] += self.pose_1_number / 30
                data_info["pose_2"] += self.pose_2_number / 30
                data_info["pose_3"] += self.pose_3_number / 30
                data_info["pose_4"] += self.pose_4_number / 30
                self.pose_0_number = 0
                self.pose_1_number = 0
                self.pose_2_number = 0
                self.pose_3_number = 0
                self.pose_4_number = 0
                torch.save(data_info, self.personal_root_list[self.user_index])

            return frame, "当前坐姿为{}".format(self.pose_list[output_class])
        except:
            return frame, "当前坐姿为Nan"

    def face_detect_function(self, frame):
        confidence_list = [0 for _ in range(len(self.names))]
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.detector(img_gray, 0)
        self.cv2_recognize_root_list = [os.path.join(self.cv2_recognize_root, each) for each in
                                        os.listdir(self.cv2_recognize_root)]
        try:
            if len(faces) <= 0:
                return frame, "正在处理信息中..."
            elif len(self.face_detect_list) <= 10:
                places = self.get_faces(faces)
                self.face_detect_list.append(img_gray[places[1]: places[3], places[0]: places[2]])
                return frame[places[1]: places[3], places[0]: places[2], :], "正在处理信息中..."
            else:
                with open(self.info_path, 'w', encoding="utf-8") as f:
                    f.write("0")
                    self.function = "0"
                for k in range(len(self.cv2_recognize_root_list)):
                    cv2_recognizer = cv2.face.LBPHFaceRecognizer.create()
                    cv2_recognizer.read(self.cv2_recognize_root_list[k])
                    idNum, confidence = cv2_recognizer.predict(self.face_detect_list[5])
                    print(confidence)
                    confidence_list[k] = confidence
                user_id = confidence_list.index(min(confidence_list))
                self.user_index = user_id
                user_name = self.names[user_id]
                self.face_detect_list = []
                self.user = user_name
                self.make_voice = True
                return self.default_image, "您好,{}".format(user_name)
        except:
            with open(self.info_path, 'w', encoding="utf-8") as f:
                f.write("0")
                self.function = "0"
            return self.default_image, "something wrong, please try again"

    def face_detect_in_function(self, frame):
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.detector(img_gray, 0)
        cv2_recognizer = cv2.face.LBPHFaceRecognizer.create()
        try:
            if len(faces) <= 0:
                return frame, "正在处理信息中..."
            elif len(self.face_detect_list) <= 30:
                places = self.get_faces(faces)
                save_img = img_gray[places[1]: places[3], places[0]: places[2]]
                self.face_detect_list.append(save_img)
                # opencv_face_saver(save_img, len(self.face_detect_list))
                return frame[places[1]: places[3], places[0]: places[2], :], "正在处理信息中..."
            else:
                with open(self.info_path, 'w', encoding="utf-8") as f:
                    f.write("0")
                    self.function = "0"
                user_number = self.face_config["number"]
                self.user = self.names[user_number]
                self.face_config.add_item({"number": user_number + 1})
                ids = np.array([user_number for _ in range(len(self.face_detect_list))])
                cv2_recognizer.train(self.face_detect_list, ids)
                save_root = os.path.join(self.cv2_recognize_root, "number{}.yml".format(user_number))
                cv2_recognizer.save(save_root)
                self.face_detect_list = []
                torch.save(self.face_config, self.config_['face_config_path'])
                dt = {"tired_number": 0, "pose_0": 0, "pose_1": 0, "pose_2": 0, "pose_3": 0, "pose_4": 0}
                save_root = r"C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints\personal_info\person_{}.pt".format(user_number)
                torch.save(dt, save_root)
                speakout("成功录入信息")
                return self.default_image, "您好,{}".format(self.user)
        except:
            with open(self.info_path, 'w', encoding="utf-8") as f:
                f.write("0")
                self.function = "0"
            return self.default_image, "something wrong, please try again"

    def default_function(self):
        if self.make_voice:
            data_info = torch.load(self.personal_root_list[self.user_index])
            speakout("您好，" + self.names[self.user_index])
            pose_list = []
            pose_list.append(data_info["pose_0"])
            pose_list.append(data_info["pose_1"])
            pose_list.append(data_info["pose_2"])
            pose_list.append(data_info["pose_3"])
            pose_list.append(data_info["pose_4"])
            max_index = pose_list.index(max(pose_list))
            speakout(self.word_list[max_index])

        return self.default_image, "您好,{}".format(self.user)

    def final_function(self):
        camera = cv2.VideoCapture(0)

        while True:
            with open(self.info_path, 'r', encoding="utf-8") as f:
                self.function = f.read().strip()
            ret, frame = camera.read()
            if frame is None:
                return self.default_image, "您好,{}".format(self.user)
            b, g, r = cv2.split(frame)
            frame = cv2.merge([r, g, b])
            if self.function == "0":
                yield self.default_function()
                time.sleep(0.2)
            elif self.function == "1":
                yield self.face_detect_in_function(frame)
            elif self.function == "2":
                yield self.face_detect_function(frame)
            elif self.function == "3":
                yield self.pose_position(frame)
            elif self.function == "4":
                yield self.tired_function(frame)
            elif self.function == "5":
                root = r"C:\Users\PokeBot\Desktop\PycharmProjects\self_introduction"
                ps = [os.path.join(root, each)for each in os.listdir(root)]
                for each in ps:
                    chunk = 1024
                    wf = wave.open(each, 'rb')
                    p = pyaudio.PyAudio()
                    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                                    rate=wf.getframerate(), output=True)
                    data = wf.readframes(chunk)  # 读取数据
                    while data != b'':  # 播放
                        stream.write(data)
                        data = wf.readframes(chunk)
                    stream.stop_stream()  # 停止数据流
                    stream.close()
                    p.terminate()
                with open(self.info_path, 'w', encoding='utf-8') as f:
                    f.write("0")
                yield self.default_function()
            else:
                yield self.default_function()

if __name__ == "__main__":
    # 一个线程监听输入，根据输入改变class中的值
    # 只需要修改config中的文件即可
    config_ = ConfigSet(face_config_path=r"C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints\face_info.pt",
                        sess_path=r"C:\Users\PokeBot\Desktop\PycharmProjects\humen_pose\movenet\single-pose-thunder"
                                  r"\model.onnx",
                        classification_path=r"C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints"
                                            r"\human_pose_pytorch\classification.onnx",
                        default_img_path=r"C:\Users\PokeBot\Desktop\PycharmProjects\MyDearMoments.jfif",
                        pointer_path=r"C:\Users\PokeBot\Desktop\PycharmProjects\shape_predictor_68_face_landmarks.dat",
                        beta=0.95,
                        path=r"C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints\test.txt.txt",
                        tired_root=r"C:\Users\PokeBot\Desktop\lcd_cmd.txt",
                        pose_root=r"C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints\pose_info.txt",
                        cv2_root=r"C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints\face_detect_saver",
                        cv2_recognize_root=r"C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints"
                                           r"\face_detect_saver",
                        personal_root=r"C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints\personal_info")
    my_function = MyFunction(config=config_)

    gui = generate_gui(input_function=my_function.final_function)

    def run_gui(gui_):
        gui_.launch()

    def write_function():
        while True:
            dt = input()
            with open(config_['path'], 'w', encoding='utf-8') as f:
                f.write(dt)


    thread2 = threading.Thread(target=write_function)
    thread1 = threading.Thread(target=run_gui, args=(gui, ))
    thread3 = threading.Thread(target=ivw_wakeup)
    thread3.start()
    thread1.start()
    thread2.start()



