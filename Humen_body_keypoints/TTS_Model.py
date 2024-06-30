from win32com import client


if __name__ == "__main__":
    eg = client.Dispatch("SAPI.Spvoice")
    eg.Rate = 5
    eg.Voice = eg.GetVoices().Item()
    print(eg.Voice)
    eg.Speak("您现在已经很疲劳了，请休息一下")