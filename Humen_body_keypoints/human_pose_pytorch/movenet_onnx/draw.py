import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.patches as patches
from keypoint import keypoints_and_edges_for_display
import time
import cv2
import torch


def cv2_draw(image, keypoints_with_scores, output_class):
    label_list = ["Normal posture", "Over Sticking", "paralyzed", "legs crossed", "hunchbacked"]
    output_class = torch.tensor(output_class)[0]

    label = torch.argmax(output_class, dim=-1)
    value = output_class[label]
    height, width, channel = image.shape
    (keypoint_locs, keypoint_edges, edge_colors) = keypoints_and_edges_for_display(
                                                                keypoints_with_scores, height, width)
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
    cv2.putText(image, text, (min_x, int(min_y - 0.05 * min_y)), cv2.FONT_HERSHEY_SIMPLEX,  2,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def draw_prediction_on_image(
        image, keypoints_with_scores, crop_region=None, close_figure=False,
        output_image_height=None):
    """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
    start_time = time.time()
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borde
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')

    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

    (keypoint_locs, keypoint_edges,
     edge_colors) = keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
        scat.set_offsets(keypoint_locs)

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        rect = patches.Rectangle(
            (xmin, ymin), rec_width, rec_height,
            linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    draw_time = time.time()
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    draw_end_time = time.time()
    # if output_image_height is not None:
    #   output_image_width = int(output_image_height / height * width)
    #   image_from_plot = cv2.resize(
    #       image_from_plot, dsize=(output_image_width, output_image_height),
    #        interpolation=cv2.INTER_CUBIC)
    print("point info process time={}".format(draw_time - start_time))
    print("draw on pic time={}".format(draw_end_time - draw_time))

    return image_from_plot