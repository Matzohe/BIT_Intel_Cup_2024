#语音唤醒所需的包
import os
from ctypes import cdll, byref, string_at, c_void_p, CFUNCTYPE, c_char_p, c_uint64, c_int64
import pyaudio
from loguru import logger
#蜂鸣器所掉的包
import winsound

#whisper所需的包
import whisper
import zhconv
import wave  # 使用wave库可读、写wav类型的音频文件
import pyaudio  # 使用pyaudio库可以进行录音，播放，生成wav文件
import pyttsx3


# 唤醒成功后打印的日志并调用whisper
def py_ivw_callback(sessionID, msg, param1, param2, info, userDate):
    # typedef int( *ivw_ntf_handler)( const char *sessionID, int msg, int param1, int param2, const void *info, void *userData );
    # 在此处编辑唤醒后的动作
    winsound.Beep(5000, 100)
    print("sessionID =>", sessionID)
    print("msg =>", msg)
    print("param1 =>", param1)
    print("param2 =>", param2)
    print("info =>", info)
    print("userDate =>", userDate)
    engine = pyttsx3.init()  # 例化一个对象，用来文本转语音
    model = whisper.load_model("base", "cpu")  # 语音转文本
    #record(3)  # 定义录音时间，单位/s
    #result = model.transcribe("output.wav", language='Chinese', fp16=32)
    #s = result["text"]
    #s1 = zhconv.convert(s, 'zh-cn')
    #print(s1)
    while True:
        record(3)  # 定义录音时间，单位/s
        result = model.transcribe("output.wav",
                                  language='Chinese', fp16=32)
        s = result["text"]
        s1 = zhconv.convert(s, 'zh-cn')
        print(s1)
        if(("没有" in s1) or ("不需要" in s1) or ("没事" in s1) or ("再见" in s1) ):
            break
        elif (("坐姿") in s1 or "坐" in s1 or "姿" in s1 or "做" in s1):
            with open(r"C:\Users\PokeBot\PycharmProjects\Humen_body_keypoints\test.txt.txt", 'w',
                      encoding='utf-8') as f:
                f.write("3")
            speakout("坐姿检测")

        elif (("录" in s1) and "入" in s1):
            with open(r"C:\Users\PokeBot\PycharmProjects\Humen_body_keypoints\test.txt.txt", 'w',
                      encoding='utf-8') as f:
                f.write("1")
            speakout("人脸录入")
        elif ("人" in s1 or "脸" in s1 or "恋" in s1) :
            with open(r"C:\Users\PokeBot\PycharmProjects\Humen_body_keypoints\test.txt.txt", 'w',
                      encoding='utf-8') as f:
                f.write("2")
            speakout("人脸识别")
        elif (("疲" in s1) or ("劳" in s1) or ("皮" in s1)):
            with open(r"C:\Users\PokeBot\PycharmProjects\Humen_body_keypoints\test.txt.txt", 'w',
                      encoding='utf-8') as f:
                f.write("4")
            print("疲劳检测")
        else:
            speakout("请再说一次")
            print("请再说一次")


CALLBACKFUNC = CFUNCTYPE(None, c_char_p, c_uint64,
                         c_uint64, c_uint64, c_void_p, c_void_p)
pCallbackFunc = CALLBACKFUNC(py_ivw_callback)


def ivw_wakeup():
    try:
        msc_load_library = r'C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints\voice\bin\msc_x64.dll'
        app_id = '06719689'  # 填写自己的app_id
        ivw_threshold = '0:1450'
        jet_path = r'C:\Users\PokeBot\Desktop\PycharmProjects\Humen_body_keypoints\voice\bin\msc\res\ivw\wakeupresource.jet'
        work_dir = 'fo|' + jet_path
    except Exception as e:
        return e

    # ret 成功码
    MSP_SUCCESS = 0

    dll = cdll.LoadLibrary(msc_load_library)
    errorCode = c_int64()
    sessionID = c_void_p()
    # MSPLogin
    Login_params = "appid={},engine_start=ivw".format(app_id)
    Login_params = bytes(Login_params, encoding="utf8")
    ret = dll.MSPLogin(None, None, Login_params)
    if MSP_SUCCESS != ret:
        logger.info("MSPLogin failed, error code is: %d", ret)
        return

    # QIVWSessionBegin
    Begin_params = "sst=wakeup,ivw_threshold={},ivw_res_path={}".format(
        ivw_threshold, work_dir)
    Begin_params = bytes(Begin_params, encoding="utf8")
    dll.QIVWSessionBegin.restype = c_char_p
    sessionID = dll.QIVWSessionBegin(None, Begin_params, byref(errorCode))
    if MSP_SUCCESS != errorCode.value:
        logger.info("QIVWSessionBegin failed, error code is: {}".format(
            errorCode.value))
        return

    # QIVWRegisterNotify
    dll.QIVWRegisterNotify.argtypes = [c_char_p, c_void_p, c_void_p]
    ret = dll.QIVWRegisterNotify(sessionID, pCallbackFunc, None)
    if MSP_SUCCESS != ret:
        logger.info("QIVWRegisterNotify failed, error code is: {}".format(ret))
        return

    # QIVWAudioWrite
    # 创建PyAudio对象
    pa = pyaudio.PyAudio()

    # 设置音频参数
    sample_rate = 16000
    chunk_size = 1024
    format = pyaudio.paInt16
    channels = 1

    # 打开音频流
    stream = pa.open(format=format,
                     channels=channels,
                     rate=sample_rate,
                     input=True,
                     frames_per_buffer=chunk_size)

    # 开始录制音频
    logger.info("* start recording")
    ret = MSP_SUCCESS
    while ret == MSP_SUCCESS:  # 这里会一直进行监听你的唤醒，只要监听到你的唤醒就调用上面的py_ivw_callback函数打印日志
        audio_data = stream.read(chunk_size)
        audio_len = len(audio_data)
        ret = dll.QIVWAudioWrite(sessionID, audio_data, audio_len, 2)
    logger.info('QIVWAudioWrite ret =>{}', ret)
    logger.info("* done recording")

    # 关闭音频流
    stream.stop_stream()
    stream.close()

    # 终止PyAudio对象
    pa.terminate()


def record(time):  # 录音程序
    # 定义数据流块
    CHUNK = 1024  # 音频帧率（也就是每次读取的数据是多少，默认1024）
    FORMAT = pyaudio.paInt16  # 采样时生成wav文件正常格式
    CHANNELS = 1  # 音轨数（每条音轨定义了该条音轨的属性,如音轨的音色、音色库、通道数、输入/输出端口、音量等。可以多个音轨，不唯一）
    RATE = 16000  # 采样率（即每秒采样多少数据）
    RECORD_SECONDS = time  # 录音时间
    WAVE_OUTPUT_FILENAME = r"output.wav"  # 保存音频路径
    p = pyaudio.PyAudio()  # 创建PyAudio对象
    stream = p.open(format=FORMAT,  # 采样生成wav文件的正常格式
                    channels=CHANNELS,  # 音轨数
                    rate=RATE,  # 采样率
                    input=True,  # Ture代表这是一条输入流，False代表这不是输入流
                    frames_per_buffer=CHUNK)  # 每个缓冲多少帧
    print("* recording")  # 开始录音标志
    frames = []  # 定义frames为一个空列表
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):  # 计算要读多少次，每秒的采样率/每次读多少数据*录音时间=需要读多少次
        data = stream.read(CHUNK)  # 每次读chunk个数据
        frames.append(data)  # 将读出的数据保存到列表中
    print("* done recording")  # 结束录音标志

    stream.stop_stream()  # 停止输入流
    stream.close()  # 关闭输入流
    p.terminate()  # 终止pyaudio

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')  # 以’wb‘二进制流写的方式打开一个文件
    wf.setnchannels(CHANNELS)  # 设置音轨数def speakout(workText):
    wf.setsampwidth(p.get_sample_size(FORMAT))  # 设置采样点数据的格式，和FOMART保持一致
    wf.setframerate(RATE)  # 设置采样率与RATE要一致
    wf.writeframes(b''.join(frames))  # 将声音数据写入文件
    wf.close()  # 数据流保存完，关闭文件


def speakout(workText):
    engine = pyttsx3.init()
    # 获取发音人
    voices = engine.getProperty('voices')
    #for voice in voices:
    #    print('id = {} \nname = {} \n'.format(voice.id, voice.name))
    #engine.setProperty('voice', voices[2].id)
    #print(voices[2].id)
    # 这里可以获取你设备当前支持的发音，可以配置下面代码使用，我这里使用zh偷懒了

    #engine.setProperty('voice', 'zh')  # 开启支持中文

    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate)  # 控制发音语速，可以自己调

    # 设置音量  范围为0.0-1.0  默认值为1.0
    engine.setProperty('volume', 0.7)

    engine.say(workText)
    engine.runAndWait()


if __name__ == '__main__':
    ivw_wakeup()
