# Deep_Emotion_MIDI

This project is based off omarsayed7's implimentation of Deep-Emotion. @https://github.com/omarsayed7/Deep-Emotion

Install all dependencies: opencv-python, pytorch, rtmidi, then run the run.py file from the Deep_Emotion_MIDI directory. The run.py file pipes live emotion recognition data to a midi bus (change the argument in midiout.open_port(n) if necessary). The idea is to manipulate midi parameters and notes in real time using facial expression. For analysing the midi outputs, I use Logic Pro, but this should in theory work for any midi based program.
