#!/usr/bin/env python3
import wave, struct
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
from matplotlib.animation import FuncAnimation

def read_wav(filename):
	wav_file = wave.open(filename)

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
	data = []
	for i in range(n_samples):
		data.append(wav_bytes[W*Ch*i : W*Ch*(i+1)])

	# drop other channel
	if Ch == 2:
		data = [frame[:W] for frame in data]

	# unpack bytes
	fmt = {
		1: "<c",
		2: "<h",
		4: "<i"
	}
	data = [struct.unpack(fmt[W], frame)[0] for frame in data]
	#data = [sample + (1<<(8*(W-1))) for sample in data]

	return data, W*8, Hz

wav_samples, sample_bits, sample_rate = read_wav(argv[1])

fourier_min = 1
fourier_max = 1000

# plot samples
plt.plot(np.linspace(0,1,len(wav_samples)), wav_samples)
plt.ylabel("Sample")
plt.xlabel("Time (s)")
plt.title("Original signal")
plt.show()

# animate samples wrapped around origin at different frequencies
fig = plt.figure()
scatter_ax = fig.add_subplot(121, projection='polar')
scatter_ax.set_ylim([-(1<<(sample_bits-1)), (1<<(sample_bits-1))])
scatter_xd, scatter_yd = [], []
scatter_ax.scatter(scatter_xd, scatter_yd)

line_ax = fig.add_subplot(122)
line_ax.set_xlim([fourier_min, fourier_max])
line_xd, line_yd = [], []
ln, = line_ax.plot(line_xd, line_yd, color='red')


def init():
	scatter_ax.set_ylim([-(1<<(sample_bits-1)), (1<<(sample_bits-1))])
	#return ln,

def update(frame):
	scatter_xd = np.linspace(0, 2*np.pi*frame, len(wav_samples))
	scatter_yd = wav_samples

	transform_sin = np.average(np.multiply(np.sin(scatter_xd), scatter_yd))
	transform_cos = np.average(np.multiply(np.cos(scatter_xd), scatter_yd))

	scatter_ax.clear()
	scatter_ax.scatter(scatter_xd, scatter_yd, marker='.', linewidth=0, s=10)

	scatter_ax.scatter(
		[np.arctan(transform_sin/transform_cos)],
		[(np.sqrt(transform_sin**2 + transform_cos**2) * 20 - (1 << (sample_bits-1)))],
		color='red', s=50)

	scatter_ax.set_ylim([-(1<<(sample_bits-1)), (1<<(sample_bits-1))])
	scatter_ax.set_title("%.2f Hz" % frame)

	if not len(line_xd) or frame > line_xd[-1]:
		line_xd.append(frame)
		line_yd.append(transform_cos)

	line_ax.set_ylim([-1000, 1000])

	ln.set_data(line_xd, line_yd)
	#return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(fourier_min,fourier_max,1000), init_func=init)
plt.show()

# compute transform
transform_samples = 1500

transform_1 = np.zeros(transform_samples)
transform_2 = np.zeros(transform_samples)

radiuses = np.array(wav_samples)

for i in range(transform_samples):
	thetas = np.linspace(0, 2*np.pi*i, len(wav_samples))

	transform_1[i] = np.average(np.multiply(np.sin(thetas), radiuses))
	transform_2[i] = np.average(np.multiply(np.cos(thetas), radiuses))

plt.plot(np.linspace(1,1500,1500), transform_1, transform_2)
plt.xlabel("Hz")
plt.ylabel("Intensity")
plt.title("Fourier transform")
plt.show()
