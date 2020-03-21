#!/usr/bin/env python3
import wave, struct
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
from sys import argv
from matplotlib.animation import FuncAnimation
from matplotlib import animation

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
	data = np.array([struct.unpack(fmt[W], frame)[0] for frame in data])

	return data, W*8, Hz

wav_samples, sample_bits, sample_rate = read_wav(argv[1])

# downsample to 8khz for better performance
wav_samples = wav_samples[::(sample_rate // 8000)]

# plot samples
plt.plot(np.linspace(0,1,len(wav_samples)), wav_samples)
plt.ylabel("Sample")
plt.xlabel("Time (s)")
plt.title("Original signal")
plt.show()

# create subplots for animation
min_hz = 1
max_hz = 1000

fig = plt.figure(figsize=(10,6))
scatter_ax = fig.add_subplot(121, projection='polar')
scatter_xd = np.linspace(0, 2*np.pi*1, len(wav_samples))
scatter_yd = wav_samples
scatter_ax.scatter(scatter_xd, scatter_yd, marker='.', linewidth=0, s=10)

line_ax = fig.add_subplot(122)
line_xd, line_yd = [], []
ln, = line_ax.plot(line_xd, line_yd, color='red')

# animate samples wrapped around origin at different frequencies
def init():	
	scatter_ax.set_ylim([-(1<<(sample_bits-2)), (1<<(sample_bits-2))])
	line_ax.set_xlim([min_hz, max_hz])

def update(frame):
	# Scale the x values to wind the wav samples around the plot $(frame) times
	scatter_xd = np.linspace(0, 2*np.pi*frame, len(wav_samples))
	scatter_ax.clear()
	scatter_ax.set_ylim([-(1<<(sample_bits-2)), (1<<(sample_bits-2))])
	scatter_ax.scatter(scatter_xd, scatter_yd, marker='.', linewidth=0, s=10)

	# Compute a "center of mass" for the plotted points
	transform_sin = np.average(np.multiply(np.sin(scatter_xd), scatter_yd+(1<<(sample_bits-2))))
	transform_cos = np.average(np.multiply(np.cos(scatter_xd), scatter_yd+(1<<(sample_bits-2))))
	# Transform back to polar coordinates
	theta = np.arctan(transform_sin / transform_cos)
	if transform_cos < 0: theta += np.pi
	length = np.sqrt(transform_sin**2 + transform_cos**2)
	plot_scale = 20
	plot_offset = 1 << (sample_bits-2)
	scatter_ax.scatter(
		[theta], [length * plot_scale - plot_offset],
		color='red', s=50)

	scatter_ax.set_title("%.2f Hz" % frame)
	scatter_ax.legend(["wav samples","\"center of mass\""], loc="upper right")

	if not len(line_xd) or frame > line_xd[-1]:
		line_xd.append(frame)
		line_yd.append(transform_cos)

	line_ax.set_ylim([-1000, 1000])
	line_ax.legend(["x position of center"])
	ln.set_data(line_xd, line_yd)

ani = FuncAnimation(fig, 
	update, 
	frames=np.linspace(min_hz,max_hz,1000), 
	interval=50,
	repeat=False,
	blit=False,
	init_func=init)

writer = animation.ImageMagickFileWriter(fps=10)
#ani.save('im.gif',writer=writer,dpi=50)
plt.show()


# compute transform
transform_samples = 1500

transform_real = np.zeros(transform_samples)
transform_imag = np.zeros(transform_samples)

hz_range = np.linspace(1, 1000, transform_samples)

i = 0
for hz in hz_range:
	thetas = np.linspace(0, 2*np.pi*hz, len(wav_samples))
	transform_imag[i] = np.average(np.multiply(np.sin(thetas), wav_samples))
	transform_real[i] = np.average(np.multiply(np.cos(thetas), wav_samples))
	i += 1

plt.plot()
plt.plot(hz_range, transform_real, 'r', hz_range, transform_imag, 'b')
plt.xlabel("Hz")
plt.ylabel("Intensity")
plt.title("\"Fourier transform\"")
plt.legend(['real part','imaginary part'])
plt.show()
