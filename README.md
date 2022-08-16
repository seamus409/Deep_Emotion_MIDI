# Deep_Emotion_MIDI

This project is based off omarsayed7's implimentation of Deep-Emotion. 

Install all dependencies: opencv-python, pytorch, Deep-Emotion, rtmidi, then run the run.py file from the Deep_Emotion_MIDI directory. The run.py file pipes live emotion recognition data to a midi bus (change the argument in midiout.open_port(n) if necessary). The idea is to manipulate midi parameters and notes in real time using facial expression. For analysing the midi outputs, I use Logic Pro X, but this should in theory work for any program that has access to midi.
