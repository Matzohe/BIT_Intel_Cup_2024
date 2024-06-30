import dlib
import numpy as np
from scipy.spatial import distance as dist
import cv2
from imutils import face_utils


# 在提取人脸特征时，需要考虑相机模型的修正

def mouth_aspect_ratio(mouth):  # 嘴部
    D = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    E = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    F = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (D + E) / (2.0 * F)
    A = np.linalg.norm(mouth[12] - mouth[18])
    B = np.linalg.norm(mouth[14] - mouth[15])
    C = np.linalg.norm(mouth[12] - mouth[14])
    mar1 = (A + B) / (2 * C)
    return mar


def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[5])  # 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear


def get_tired_thread(eye_data, mouth_data, is_eye=True):
    return 0.3 * max(eye_data) + 0.7 * min(eye_data), 2 * min(mouth_data)


def get_fatigue_score(eye_thread, mouth_thread, eye_list, mouth_list):
    e_list = np.array(eye_list)
    m_list = np.array(mouth_list)
    rate1 = np.sum(e_list < eye_thread) / len(eye_list)
    sum2 = np.sum(m_list > mouth_thread)
    score = 15 * (sum2 > 0)
    if rate1 <= 0.5:
        score = score + 0
    elif rate1 <= 0.75:
        score = score + 3
    else:
        score = score + 10
    return score


class GetTiredScore:
    def __init__(self, beta=0.9):
        self.is_first = True
        self.tired_score = 0
        self.eye_thread = 0
        self.mouth_thread = 0
        self.beta = beta

    def get_human_face_point(self, is_first=True, ear_thread=None, mar_thread=None):
        if not is_first:
            self.is_first = False
            self.eye_thread = ear_thread
            self.mouth_thread = mar_thread
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(r"M:\github_project\Fatigue-driving-detection-system-based-on-opencv-dlib-main"
                                         r"\Fatigue driving detection system\model\shape_predictor_68_face_landmarks.dat")
        # face_utils可以将不同区域的关键点坐标和其索引相联系
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        cap = cv2.VideoCapture(0)  # 设置默认摄像头为0
        frame = 0
        mar_list = []
        ear_list = []
        while cap.isOpened():

            flag, im_rd = cap.read()

            # 将图像取灰度
            img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)

            # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数faces
            faces = detector(img_gray, 0)
            # 如果检测到人脸
            if len(faces) != 0:
                # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
                for k, d in enumerate(faces):
                    # 用红色矩形框出人脸
                    # cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)
                    # 使用预测器得到68点数据的坐标
                    shape = predictor(im_rd, d)
                    # 圆圈显示每个特征点
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
                    # 将脸部特征信息转换为数组array的格式
                    shape = face_utils.shape_to_np(shape)

                    mouth = shape[mStart:mEnd]
                    left_eye = shape[lStart: lEnd]
                    right_eye = shape[rStart: rEnd]
                    mar = mouth_aspect_ratio(mouth)
                    leftEAR = eye_aspect_ratio(left_eye)
                    rightEAR = eye_aspect_ratio(right_eye)
                    ear = (leftEAR + rightEAR) / 2.0
                    ear_list.append(ear)
                    mar_list.append(mar)
                    frame += 1
                    # 仅取第一张人脸进行处理
                    break

            else:
                print("未检测到人脸")
            if (frame + 1) % 31 == 0:
                if self.is_first:
                    self.eye_thread, self.mouth_thread = get_tired_thread(ear_list, mar_list)
                    self.is_first = False
                else:
                    self.tired_score = \
                        get_fatigue_score(self.eye_thread, self.mouth_thread, ear_list, mar_list)\
                        + self.beta * self.tired_score
                    if self.tired_score > 100:
                        self.tired_score = 100
                frame = 0
                ear_list = []
                mar_list = []
                print("现在您的疲劳度为{}".format(self.tired_score))






if __name__ == "__main__":
    run_function = GetTiredScore()
    run_function.get_human_face_point()

