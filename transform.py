import wave, struct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

wav_file = wave.open("a2002011001-e02.wav")
#wav_file = wave.open("SineWaveMinus16.wav")

W = wav_file.getsampwidth()
print(W*8, "bits")
Hz = wav_file.getframerate()
print(Hz, "hz")
Ch = wav_file.getnchannels()
print(Ch, "ch")
N = wav_file.getnframes()
print(N, "frames")

# read 20000 frames data
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
#print(wav_samples)

plt.plot(np.linspace(0,1,len(wav_samples)), wav_samples)
plt.show()

# shift up to unsigned


print(wav_samples)

fig, ax = plt.subplots()
xd, yd = [], []

wrap_hz = 1

ln, = plt.polar(np.linspace(0, 2*np.pi, len(wav_samples)), wav_samples)


def init():
    #ax.set_xlim(0, 2*np.pi)
    #ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xd = np.linspace(0, 2*np.pi*frame, len(wav_samples))
    yd = wav_samples
    ln.set_data(xd, yd)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(1,10,20),
                    init_func=init)

plt.show()

transform_samples = 2000

transform_1 = np.zeros(transform_samples)
transform_2 = np.zeros(transform_samples)

radiuses = np.array(wav_samples)

for i in range(transform_samples):
	print(i)
	thetas = np.linspace(0, 2*np.pi*i, len(wav_samples))

	sines = np.sin(thetas)
	cosines = np.cos(thetas)
	# which is imaginary part?
	sines = np.multiply(sines, radiuses)
	cosines = np.multiply(cosines, radiuses)

	transform_1[i] = np.average(sines)
	transform_2[i] = np.average(cosines)

plt.plot(np.linspace(1,2000,2000), transform_1, transform_2)
plt.show()



