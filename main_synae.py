#def main_synae():
import math
import numpy as np
import matplotlib.pyplot as mp
import wave
import pyaudio
import scipy.misc
import itertools

#import vec2frames
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

signal = [a*b for a,b in zip(signal, np.hanning(L))] #multiply with hann wnd to smooth edges
arr = np.array(signal)
vec=arr/max(arr) #normalise

Tw = 32    # analysis frame duration (ms)
Ts = Tw/8  # analysis frame shift (ms)
#
Nw = round( fs*Tw*0.001 )  # frame duration (in samples)
Ns = round( fs*Ts*0.001 )  # frame shift (in samples)
nfft = 2^nextpow2( 2*Nw )  # FFT analysis length
# nfft= Nw
# divide signal into frames {frames, indexes = vec2frames( vec, Nw, Ns)}

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
indf=[Ns*i for i in range(0,M)]# indexes for frames
inds=[i for i in range(0,Nw)]# indexes for samples
indexes = np.tile(indf,[Nw,1]) + list(zip(*np.tile(inds,[M,1])))# combined framing indexes
Frames = vec[indexes]

window = np.hanning( Nw )

frames = np.dot(list(zip(*Frames)), np.diag(window))

# perform short-time Fourier transform (STFT) analyses
X = np.fft.fft( frames )
d1, d2 = len(X), len(X[0])
t = np.linspace(0, Tdur, TdurSamp)
freq = np.fft.fftfreq(len(frames[0]), t[1]-t[0])

# mp.figure()
# mp.plot(freq, np.abs(X[0]))
# mp.figure()
# mp.plot(freq, np.angle(X[0]))
# mp.show()

################################# get Gains from Image
# I2=double(I(:,:,1))./255;
# I2r=imresize(I2',[size(X,1) size(X,2)/2]);%imshow(I2)
# MASK=[I2r(:,end:-1:1) I2r];
I0 = scipy.misc.imread('C:\\Users\\Stefano\\Pictures\\io_BW.png', [True, 'F'])
I2 = scipy.misc.imresize(I0, [d2, round( d1 )], interp='bilinear', mode=None)
MASK = np.column_stack( list(reversed(I2)) )
# MASK = np.column_stack( (I2, list(reversed(I2))) )
# mp.imshow(MASK, cmap=mp.cm.gray)
# mp.show()


#Y = MASK .* exp(1i*angle(X));
Y = MASK * np.exp(1j*np.angle(X))

# apply inverse STFT
Yframes = np.real( np.fft.ifft(Y,nfft) )
#
# % discard FFT padding from frames
framesy = Yframes[:,:Nw]
#
# % perform overlap-and-add synthesis
# y = frames2vec( frames.y, indexes, 'rows', @hanning, 'G&L' );
#
vec, wsum = np.zeros(L), np.zeros(L)

# Griffin & Lim's method
# overlap-and-add syntheses,
frames = np.dot(framesy, np.diag(window))
# overlap-and-add frames
for m in range(M-1):
    vec[indexes[:,m]] = vec[indexes[:,m]] + frames[m,:]
# overlap-and-add window samples
for m in range(M-1):
    wsum[indexes[:,m]] = wsum[indexes[:,m]] + window**2
# # for some tapered analysis windows, use:
wsum[wsum<1E-2] = 1E-2
#
# # divide out summed-up analysis windows
vec = vec / wsum
# vec[abs(vec)>vec.mean()+1.5*vec.std()]=vec.mean()+3.5*vec.std()
vec=vec/vec.max()
# % truncate extra padding (back to original signal length)
y = np.multiply(vec[:L], np.hanning(L))
#
mp.figure()
mp.subplot(1, 2, 1)
mp.plot(t, y)

mp.subplot(1, 2, 2)
# mp.imshow(MASK)
mp.specgram(y, Fs = fs)

mp.show()
# sound(y,fs)
# %
# t=linspace(0,Tdur,Tdur*fs);
# figure(1);
# subplot(121); plot(t,x,'k',t,y,'r'); ylim([-1.5 1.5])
# % subplot(121); imshow(MASK);
# subplot(122); spectrogram(y,4*Tw,Tw,nfft,fs,'yaxis');