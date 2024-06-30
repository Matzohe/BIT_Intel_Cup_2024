import cv2
import io
from io import BytesIO
import gradio as gr


def video_io(function=None):
    def get_camera_input():
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            for i in range(3000):

                ret, frame = camera.read()
                b, g, r = cv2.split(frame)
                frame = cv2.merge([r, g, b])
                if ret:
                    yield frame
    if function is None:
        function = get_camera_input
    gui = gr.Interface(
        fn=function,
        inputs=None,
        outputs=gr.Image(label="video"),
        live=True,
    )
    return gui


if __name__ == "__main__":
    gui = video_io()
    gui.launch()