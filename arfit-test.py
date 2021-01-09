#!/usr/bin/env python3
# testing script for ARfit utility
# one can either input 
# sample usage:
#   python3 arfit-test.py sample.wav 1 40

import arfit
import librosa
import sys
from python_speech_features import sigproc

def load_frames(in_file, srate=16000):
    '''load frames from either single wav or npy
    @in_file: input wav or npy
    @return: raw frames in the size of [num_frames, frame_len]
    '''
    signal, srate = librosa.load(in_file, srate)
    pre_emphed = sigproc.preemphasis(signal, coeff=0.95)
    frames = sigproc.framesig(pre_emphed, 0.025*srate, 0.01*srate)
    return frames


def main():
    frames = load_frames(sys.argv[1])
    min_ar_order = int(sys.argv[2])
    max_ar_order = int(sys.argv[3])
    ar_order = arfit.arfit_frames(frames, pmin=min_ar_order, pmax=max_ar_order)
    print(ar_order)

if __name__ == "__main__":
    main()
