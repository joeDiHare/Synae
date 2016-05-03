#def main_synae():
import math
import numpy as np
import matplotlib.pyplot as mp
import wave
import pyaudio

#import vec2frames
import time
mp.close("all")
#Find 2^n that is equal to or greater than.
def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n


# url='https://drive.google.com/open?id=0B1rTl-OBepL0MEJ4Um9JRk40MG8'
# hrs = {"User-Agent":
#        "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Chrome/16.0.912.63 Safari/535.7"}
# request = urllib.request.Request(url, headers = hrs)
# page = urllib.request.urlopen(request)
# fname = "Play"+str(int(time.time()))+".wav"
# file = open(fname, 'wb')
#define stream chunk
chunk = 1024
#open a wav format music
fname = "C:\\Users\\Stefano\\Music\\x1.wav"
f = wave.open(fname,"rb")
#instantiate PyAudio
p = pyaudio.PyAudio()
#open stream
stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                channels = f.getnchannels(),
                rate = f.getframerate(),
                output = True)
#read data
# data = f.readframes(chunk)
# #play stream
# while data != '':
#        stream.write(data)
#        data = f.readframes(chunk)
# #stop stream
# stream.stop_stream()
# stream.close()

#close PyAudio
p.terminate()
# wave.open(file, 'rb')
signal = f.readframes(-1)
signal = np.fromstring(signal, 'Int16')

# [x,fs]=wavread('x1');% x=randn(1,fs*Tdur);
fs=20e3
Tdur=1.5 #%sec
TdurSamp = round(Tdur*fs)
signal = list(reversed(signal[0:TdurSamp]))
# arr = np.array(signal); reversed_arr = arr[::-1]
L=len(signal)
Time=np.linspace(0, len(signal)/fs, num=L)

signal = [a*b for a,b in zip(signal, np.hanning(L))] #multiply with hann wnd to smooth edges
arr = np.array(signal)
vec=arr/max(arr) #normalise


#mp.figure(1)
#mp.plot(Time, vec, 'r')
#mp.show()
Tw = 32    # analysis frame duration (ms)
Ts = Tw/8  # analysis frame shift (ms)
#
Nw = round( fs*Tw*0.001 )  # frame duration (in samples)
Ns = round( fs*Ts*0.001 )  # frame shift (in samples)
nfft = 2^nextpow2( 2*Nw )  # FFT analysis length

# divide signal into frames
#frames, indexes = vec2frames( vec, Nw, Ns)

M = math.floor((L-Nw)/Ns+1)             # number of frames

# figure out if the input vector can be divided into frames exactly
E = (L-((M-1)*Ns+Nw))

# see if padding is actually needed
if( E>0 ):
    # how much padding will be needed to complete the last frame?
    P = Nw-E
    # pad with zeros
    vec[len(vec):] = [np.zeros(P)]  # pad with zeros
    M = M+1  # increment the frame count
# else:# if not padding required, decrement frame count (not very elegant solution)
#     M = M-1

# compute index matrix in the direction ='rows'
indf = np.tile(np.array((range(0, Nw))) , [M-1,1])         # indexes for frames
# inds = list(range(0, Nw-1))                              # indexes for samples
# indexes = indf[:,np.ones(1,Nw)] + inds[np.ones(M,1),:]       # combined framing indexes
# frames = vec( indexes ) # divide the input signal into frames using indexing
frames = [vec[i] for i in indf]

window = np.hanning( Nw )

frames = np.dot(frames, np.diag( window ) )

# perform short-time Fourier transform (STFT) analyses
X = np.fft.fft( frames )
t = np.linspace(0, Tdur, TdurSamp)
freq = np.fft.fftfreq(len(frames[0]), t[1]-t[0])

# mp.figure()
# mp.plot(freq, np.abs(X[0]))
# mp.figure()
# mp.plot(freq, np.angle(X[0]))
# mp.show()
#
# % get Gains from Image
# I=imread('C:\Users\joeDiHare\Pictures\io_BW.png');
# I2=double(I(:,:,1))./255;
# I2r=imresize(I2',[size(X,1) size(X,2)/2]);%imshow(I2)
# MASK=[I2r(:,end:-1:1) I2r];
# %MASK=[I2r(end:-1:1,:); I2r(1:1:end,:)];


