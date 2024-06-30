# import os
# import wave
# import pyaudio
#
# root = r"C:\Users\PokeBot\Desktop\PycharmProjects\self_introduction"
# ps = [os.path.join(root, each) for each in os.listdir(root)]
#
# for each in ps:
#     chunk = 1024
#     wf = wave.open(each, 'rb')
#     p = pyaudio.PyAudio()
#     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
#                     rate=wf.getframerate(), output=True)
#     data = wf.readframes(chunk)  # 读取数据
#     while data != b'':  # 播放
#         stream.write(data)
#         data = wf.readframes(chunk)
#         print('while循环中！')
#         print(data)
#     stream.stop_stream()  # 停止数据流
#     stream.close()
#     p.terminate()


import torch
dt = {"tired_number": 0, "pose_0": 0, "pose_1": 0, "pose_2": 0, "pose_3": 0, "pose_4": 0}
save_root = r"C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints\personal_info\person_2.pt"
torch.save(dt, save_root)