# arfit-python

This is a pure-python intepretation of [ARfit toolkit](https://github.com/tapios/arfit), based on [numpy](https://numpy.org) (so far)

### Notes
This script so far only works for 1-dimensional time series, such as wav files for example. Therefore, for multi-variate data it is recommeneded to check on MATLAB scripts in the Github link above. I will update those accordingly in near future.

### Usage
Please check `arfit-test.py` for running a test with wav file as input. 
By default we assume sample rate is 16KHz and window size and step are 25ms/10ms respectively. If not, please modify the corresponding variables in the script. Hope that is self-explained well enough.

One can check:
    * [python_speech_features](https://github.com/jameslyons/python_speech_features) for creating windowed frames;
    * [librosa](https://librosa.org) for loading single wav files.

### TODO
Needs to be planned more. I wanna extend the toolkit itself with more fancy ideas and not all utilities are useful for at least my analysis.

I also would like to have a Rust version of this but I need to investigate if numpy is supported well enough in such language. There is one as I know...