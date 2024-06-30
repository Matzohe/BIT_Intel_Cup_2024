import numpy as np


def compute_singular_value(img):
    # 在多通道图像中，将三个维度连接起来后计算相关的奇异值。
    h, w, c = img.shape
    img = img.reshape(h, w*c)
    u, sigma, v = np.linalg.svd(img)
    return u, sigma, v


def compute_similarity(img, info_tuple, select_number):
    u, sigma, v = info_tuple
    lambda_list = []
    h, w, c = img.shape
    img = img.reshape(h, w*c)
    for i in range(select_number):
        ui = u[:, i]
        vi = v[i, :]
        lambda_list.append(ui.T.dot(img).dot(vi))
    lambda_array = np.array(lambda_list)
    distance = np.linalg.norm(lambda_array - sigma[: select_number])
    return lambda_list, distance
