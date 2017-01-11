import speech_recognition as sr

def authentication(message,compare):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(message)
        audio = r.listen(source)
    try:
        codeWord = r.recognize_google(audio)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return False
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return False
    if (codeWord == compare):
        return True
    else:
        return False


while(authentication("Expected log in info...","tom") != True):
    print("unknown user")
while(authentication("password","hello world") != True):
    print("incorrect password")
