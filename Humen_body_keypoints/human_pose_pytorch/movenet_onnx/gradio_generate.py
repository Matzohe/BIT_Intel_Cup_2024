import gradio as gr


def gradio_generate(model):
    iface = gr.Interface(fn=model, inputs=gr.Image(label="上传图片", type='filepath'), outputs=gr.Image(label="处理后图片"),
                         title="坐姿检测")
    iface.launch()