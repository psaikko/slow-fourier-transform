#!/usr/bin/env python3
import wave, struct
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
from matplotlib.animation import FuncAnimation

wav_file = wave.open(argv[1])

# read metadata
W = wav_file.getsampwidth()
print(W*8, "bits")
Hz = wav_file.getframerate()
print(Hz, "hz")
Ch = wav_file.getnchannels()
print(Ch, "ch")
N = wav_file.getnframes()
print(N, "frames")

# read 1 second of data
n_samples = Hz
wav_bytes = wav_file.readframes(n_samples)

# split by sample
wav_frames = []
for i in range(n_samples):
	wav_frames.append(wav_bytes[W*Ch*i : W*Ch*(i+1)])

# drop other channel
if Ch == 2:
	wav_frames = [frame[:W] for frame in wav_frames]

# unpack bytes
fmt = {
	1: "<c",
	2: "<h",
	4: "<i"
}
wav_samples = [struct.unpack(fmt[W], frame)[0] for frame in wav_frames]
wav_samples = [sample + (1<<(8*(W-1))) for sample in wav_samples]

# plot samples
plt.plot(np.linspace(0,1,len(wav_samples)), wav_samples)
plt.ylabel("Sample")
plt.xlabel("Time (s)")
plt.title("Original signal")
plt.show()

# animate samples wrapped around origin at different frequencies
fig, ax = plt.subplots(1,1,subplot_kw=dict(projection='polar'))
xd, yd = [], []
ln, = plt.polar(np.linspace(0, 2*np.pi, n_samples), wav_samples)

def init():
	return ln,

def update(frame):
	xd = np.linspace(0, 2*np.pi*frame, n_samples)
	yd = wav_samples
	ln.set_data(xd, yd)
	ax.set_ylim([0,1<<(W*8)])
	plt.title("%.2f Hz" % frame)
	return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(1,100,1000), init_func=init)
plt.show()

# compute transform
transform_samples = 1500

transform_1 = np.zeros(transform_samples)
transform_2 = np.zeros(transform_samples)

radiuses = np.array(wav_samples)

for i in range(transform_samples):
	thetas = np.linspace(0, 2*np.pi*i, n_samples)

	transform_1[i] = np.average(np.multiply(np.sin(thetas), radiuses))
	transform_2[i] = np.average(np.multiply(np.cos(thetas), radiuses))

plt.plot(np.linspace(1,1500,1500), transform_1, transform_2)
plt.xlabel("Hz")
plt.ylabel("Intensity")
plt.title("Fourier transform")
plt.show()
