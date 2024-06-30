import torch
import onnx
from scipy.spatial import distance as dist
import numpy as np
import onnxruntime
from imutils import face_utils
import cv2
import torch.nn.functional as F


KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (255, 0, 255),
    (0, 2): (0, 255, 255),
    (1, 3): (255, 0, 255),
    (2, 4): (0, 255, 255),
    (0, 5): (255, 0, 255),
    (0, 6): (0, 255, 255),
    (5, 7): (255, 0, 255),
    (7, 9): (255, 0, 255),
    (6, 8): (0, 255, 255),
    (8, 10): (0, 255, 255),
    (5, 6): (255, 255, 0),
    (5, 11): (255, 0, 255),
    (6, 12): (0, 255, 255),
    (11, 12): (255, 255, 0),
    (11, 13): (255, 0, 255),
    (13, 15): (255, 0, 255),
    (12, 14): (0, 255, 255),
    (14, 16): (0, 255, 255)
}


def keypoints_and_edges_for_display(keypoints_with_scores,
                                    height,
                                    width,
                                    keypoint_threshold=0.11):
    """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances, _, _, _ = keypoints_with_scores.shape
    for idx in range(num_instances):
        kpts_x = keypoints_with_scores[0, idx, :, 1]
        kpts_y = keypoints_with_scores[0, idx, :, 0]
        kpts_scores = keypoints_with_scores[0, idx, :, 2]
        kpts_absolute_xy = np.stack(
            [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
        kpts_above_thresh_absolute = kpts_absolute_xy[
                                     kpts_scores > keypoint_threshold, :]
        keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                    kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
    if keypoints_all:
        keypoints_xy = np.concatenate(keypoints_all, axis=0)
    else:
        keypoints_xy = np.zeros((0, 17, 2))

    if keypoint_edges_all:
        edges_xy = np.stack(keypoint_edges_all, axis=0)
    else:
        edges_xy = np.zeros((0, 2, 2))
    return keypoints_xy, edges_xy, edge_colors


def cv2_image_process(image):
    new_image = cv2.resize(image, (256, 256))
    new_image = np.expand_dims(new_image, axis=0).astype(np.int32)
    display_image = image
    return new_image, display_image


def pose_detect_show(frame, sess, classification, openvino_model=True):
    middle_image, frame = cv2_image_process(frame)
    if openvino_model is True:
        # sess_output_layer = sess.output(0)
        # sess_input_layer = sess.input(0)
        classification_input_layer = classification.input(0)
        classification_output_layer = classification.output(0)
        # request = sess.create_infer_request()
        classify_request = classification.create_infer_request()
        # request.infer(inputs={sess_input_layer.any_name: middle_image})

        # openvino_output = request.get_output_tensor(sess_output_layer.index).data
        # openvino_output = torch.tensor(openvino_output[0])
        output = sess.run(None, {'input': middle_image})
        output = torch.tensor(output[0])
        # classify = classification.run(None, {'input': openvino_output.numpy()})
        classify_request.infer(inputs={classification_input_layer.any_name: output})
        classify = classify_request.get_output_tensor(classification_output_layer.index).data

        openvino_new_classify = torch.from_numpy(classify[0]).argmax(dim=-1)
        output_image = cv2_draw(frame, output, classify)
        return output_image, output, openvino_new_classify
    # 这里需要按照需求进行处理，能够返回三部分内容
    else:
        output = sess.run(None, {'input': middle_image})
        output = torch.tensor(output[0])
        classify = classification.run(None, {'input': output.numpy()})
        new_classify = torch.from_numpy(classify[0]).argmax(dim=-1)
        output_image = cv2_draw(frame, output, classify[0])
        return output_image, output, new_classify


def compute_singular_value(img):
    # 在多通道图像中，将三个维度连接起来后计算相关的奇异值。
    h, w, c = img.shape
    img = img.reshape(h, w * c)
    u, sigma, v = np.linalg.svd(img)
    return u, sigma, v


def compute_similarity(img, info_tuple, select_number):
    u, sigma, v = info_tuple
    lambda_list = []
    h, w, c = img.shape
    img = img.reshape(h, w * c)
    for i in range(select_number):
        ui = u[:, i]
        vi = v[i, :]
        lambda_list.append(ui.T.dot(img).dot(vi))
    lambda_array = np.array(lambda_list)
    distance = np.linalg.norm(lambda_array - sigma[: select_number])
    return lambda_list, distance


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
            each = torch.tensor(each).unsqueeze(-1).permute(2, 0, 1).unsqueeze(0)
            each = F.interpolate(each, size=(256, 256))
            each = each.squeeze(0).permute(1, 2, 0).numpy()
            avg_img += each / len(img_list)
        if len(img_list) == 0:
            raise ValueError("没有保存任何包含人脸的图像，请检查摄像头输入")
        singular_value = compute_singular_value(avg_img)
        face_info_dict[human_id] = singular_value
        config_.add_item({"face_info_dict": face_info_dict})
        return human_id


def face_test(img, config):
    simi_info = []
    each = torch.tensor(img).unsqueeze(-1).permute(2, 0, 1).unsqueeze(0)
    each = F.interpolate(each, size=(256, 256))
    img = each.squeeze(0).permute(1, 2, 0).numpy()
    for key, value in config["face_info_dict"].items():
        _, distance = compute_similarity(img, value, select_number=128)
        simi_info.append(distance)
    ts = torch.tensor(simi_info)
    idx = int(torch.argmax(ts, dim=-1))
    person_name = "S_" + str(idx)
    trust_value = ts[idx]
    return person_name, idx, trust_value


def cv2_draw(image, keypoints_with_scores, output_class):
    if keypoints_with_scores.shape[0] == 0:
        return image
    label_list = ["Normal posture", "Over Sticking", "paralyzed", "legs crossed", "hunchbacked"]

    output_class = torch.tensor(output_class)[0]

    label = torch.argmax(output_class, dim=-1)

    value = output_class[label]
    height, width, channel = image.shape

    (keypoint_locs, keypoint_edges, edge_colors) = keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    if keypoint_locs.shape[0] == 0:
        return image

    # 使用circle函数画点
    ct = [(int(keypoint_locs[i, 0]), int(keypoint_locs[i, 1])) for i in range(keypoint_locs.shape[0])]

    min_x = int(min(keypoint_locs[:, 0]) * 0.95)
    max_x = int(max(keypoint_locs[:, 0]) * 1.05)
    min_y = int(min(keypoint_locs[:, 1]) * 0.95)
    max_y = int(max(keypoint_locs[:, 1]) * 1.05)
    start1 = keypoint_edges[:, 0, :].astype(np.int32)
    start2 = keypoint_edges[:, 1, :].astype(np.int32)

    # start1 = [(keypoint_edges[i, 0, 0], keypoint_edges[i, 0, 1]) for i in range(keypoint_edges.shape[0])]
    # start2 = [(keypoint_edges[i, 1, 0], keypoint_edges[i, 1, 1]) for i in range(keypoint_edges.shape[0])]
    for each in ct:
        cv2.circle(image, center=each, radius=5, color=(255, 20, 147))
    for i in range(keypoint_edges.shape[0]):
        cv2.line(image, start1[i], start2[i], edge_colors[i], 4)
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color=(255, 0, 0), thickness=2)
    text = label_list[label] + " {}".format(value)
    cv2.putText(image, text, (min_x, int(min_y - 0.05 * min_y)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def mouth_aspect_ratio(mouth):  # 嘴部
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    # A = np.linalg.norm(mouth[12] - mouth[18])
    # B = np.linalg.norm(mouth[14] - mouth[15])
    # C = np.linalg.norm(mouth[12] - mouth[14])
    # mar1 = (A + B) / (2 * C)
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


def get_tired_thread(eye_data, mouth_data):
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


def get_human_face_point(frame, detector, predictor):
    # face_utils可以将不同区域的关键点坐标和其索引相联系
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    mar_list = []
    ear_list = []
    # 将图像取灰度
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 使用人脸检测器检测每一帧图像中的人脸。并返回人脸数faces
    faces = detector(img_gray, 0)
    # 如果检测到人脸
    if len(faces) != 0:
        # enumerate方法同时返回数据对象的索引和数据，k为索引，d为faces中的对象
        for k, d in enumerate(faces):
            # 用红色矩形框出人脸
            cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 1)
            # 使用预测器得到68点数据的坐标
            shape = predictor(frame, d)
            # 圆圈显示每个特征点
            for i in range(68):
                cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 2, (0, 255, 0), -1, 8)
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
        return frame, ear_list[0], mar_list[0]
    else:
        return frame, -1, -1


def opencv_face_saver(face_img, num):
    save_root = r"C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints\face_detect_saver\number_{}.jpg".format(num)
    cv2.imwrite(save_root, face_img)
