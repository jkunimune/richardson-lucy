"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from imageio.v2 import mimsave
from imageio.v3 import imread
from numpy.typing import NDArray
from scipy import signal


FRAME_DURATION = 0.7
RESOLUTION = 100
TRUTH_COLOR = "#ff9444"
FIT_COLOR = "#7b0031"


def main():
	np.random.seed(1)
	os.makedirs("results", exist_ok=True)

	# define the source
	x_source = np.linspace(0, 2.5, 101)
	y_source = 500*(
			bell_curve(x_source, 0.3, 0.6, 4) +
			bell_curve(x_source, 0.7, 0.6, 2) +
			shoe_curve(x_source, 2.2, 0.6, -0.5, 3)
	)

	# define the point-spread function
	x_kernel = np.linspace(0, 5.0, 201)
	Δx = x_kernel[1] - x_kernel[0]
	y_kernel = shoe_curve(x_kernel, 0.2, 0.4, +0.6, 1)

	# define the point-spread function as a matrix
	source_to_image = np.transpose(
		[signal.convolve(y, y_kernel, mode="full")*Δx for y in np.eye(x_source.size)]
	)
	source_to_data = (source_to_image[0:-1] + source_to_image[1:])/2
	source_to_image = source_to_image/np.sum(source_to_data, axis=0, keepdims=True)
	source_to_data = source_to_data/np.sum(source_to_data, axis=0, keepdims=True)

	# generate the data
	x_image = np.concatenate([x_kernel[:-1], x_kernel[-1] + x_source])
	y_data = np.random.poisson(source_to_data @ y_source)

	# set up the plotting axes
	fig = plt.figure(facecolor="none", figsize=(5.0, 2.5))
	grid = fig.add_gridspec(2, 2)
	source_ax = fig.add_subplot(grid[0, 0])
	kernel_ax = fig.add_subplot(grid[0, 1])
	image_ax = fig.add_subplot(grid[1, :])
	for ax in image_ax, kernel_ax, source_ax:
		ax.set_xticks([])
		ax.set_yticks([])

	kernel_ax.fill_between(
		x_kernel, 0, y_kernel,
		color=TRUTH_COLOR, edgecolor="none")
	kernel_ax.set_ylim(0, None)
	kernel_ax.set_title("Point-spread function")

	source_ax.fill_between(
		x_source, 0, y_source,
		color=TRUTH_COLOR, edgecolor="none")
	source_ax.set_xlim(x_source[0], x_source[-1])
	source_ax.set_ylim(0, None)
	source_ax.set_title("Object")

	image_ax.fill_between(
		np.repeat(x_image, 2)[1:-1], 0, np.repeat(y_data, 2),
		color=TRUTH_COLOR, edgecolor="none")
	image_ax.set_ylim(0, None)
	image_ax.set_title("Measurement")

	plt.tight_layout()
	source_ax_width = source_ax.get_window_extent().width
	kernel_ax_width = kernel_ax.get_window_extent().width
	kernel_ax.set_xlim(0, kernel_ax_width/source_ax_width*x_source[-1])
	image_ax_width = image_ax.get_window_extent().width
	image_ax.set_xlim(0, image_ax_width/source_ax_width*x_source[-1])

	# come up with some nice round-ish exponentially-ish increasing indices
	indices = np.concatenate([[0], np.round(1.59**np.arange(20)).astype(int)])
	for factor in [1000, 500, 100, 50, 10, 5]:
		high_enough = indices > 5*factor
		indices[high_enough] = np.round(indices[high_enough]/factor).astype(int)*factor

	# do the Richardson–lucy
	y_source_guesses = richardson_lucy(
		transfer_matrix=source_to_data,
		data=y_data,
		initial_guess=np.full(x_source.shape, np.sum(y_data)/np.sum(y_kernel)/(x_source[-1] - x_source[0])),
		num_iterations=indices[-1],
	)

	# find an appropriate stopping point
	error = np.array(
		[np.sum((y_source_guess - y_source)**2) for y_source_guess in y_source_guesses]
	)
	num_good_indices = np.argmin(error[indices]) + 1
	num_indices = indices.size

	# then do the animation part
	plt.savefig(f"results/frame-00.png", dpi=RESOLUTION)

	source_fit, = source_ax.plot(x_source, np.zeros_like(x_source), linestyle="dashed", color=FIT_COLOR, linewidth=1.5)
	image_fit, = image_ax.plot(x_image, np.zeros_like(x_image), linestyle="dashed", color=FIT_COLOR, linewidth=1.5)
	label = source_ax.text(0.99, 0.94, "",
	                       horizontalalignment="right",
	                       verticalalignment="top",
	                       transform=source_ax.transAxes)
	for i, j in enumerate(indices):
		y_image_guess = source_to_image @ y_source_guesses[j]
		source_fit.set_ydata(y_source_guesses[j])
		image_fit.set_ydata(y_image_guess)
		if j == 0:
			label.set_text(f"Initial guess")
		elif j == 1:
			label.set_text("1 iteration")
		else:
			label.set_text(f"{j} iterations")
		plt.pause(.01)
		plt.savefig(f"results/frame-{i + 1:02d}.png", dpi=RESOLUTION)

	# save a GIF that stops just past the optimal iteration
	make_gif(num_good_indices + 2, 1/FRAME_DURATION)
	# and one that goes noticeably farther
	make_gif(num_indices + 1, 1.5/FRAME_DURATION)

	plt.show()


def richardson_lucy(transfer_matrix: NDArray[float],
                    data: NDArray[float],
                    initial_guess: NDArray[float],
                    num_iterations: int) -> NDArray[float]:
	"""
	apply the Richardson–Lucy deconvolution algorithm to the given measurement and point-spread
	function, starting from the given starting point.
	:return: all the intermediate sources
	"""
	guess = initial_guess
	guesses = [guess]
	for i in range(num_iterations):
		guess = guess * (transfer_matrix.T @ (data / (transfer_matrix @ guess)))
		guesses.append(guess)
	return np.array(guesses)


def bell_curve(x: NDArray[float], x0: float, width: float, height: float) -> NDArray[float]:
	""" a bell-curve that reaches zero after a finite amonut of time """
	return height*np.where(
		x < x0 - width/2, 0, np.where(
			x < x0 - width/6, 6*((x - x0 + width/2)/width)**2, np.where(
				x < x0 + width/6, (1 - 12*((x - x0)/width)**2), np.where(
					x < x0 + width/2, 6*((x - x0 - width/2)/width)**2, 0,
				)
			)
		)
	)


def shoe_curve(x: NDArray[float], x0: float, bell_width: float, tail_length: float, height: float) -> NDArray[float]:
	""" a convolution of a bell-curve with a half-exponential """
	base = np.where((x - x0)/tail_length >= 0, np.exp(-(x - x0)/tail_length), 0)
	kernel = bell_curve(x, (x[-1] + x[0])/2, bell_width, 1)
	result = signal.convolve(base, kernel, mode="same")
	return result/np.max(result)*height


def make_gif(num_frames: int, frame_rate: float):
	""" load the images from frames/ and compile them into an animated GIF """
	# load each frame and put them in a list
	frames = []
	for i in range(num_frames):
		frame = imread(f"results/frame-{i:02d}.png")
		rgb = frame[:, :, :3]
		alpha = frame[:, :, 3, np.newaxis]/255.
		frame = (rgb*alpha + 255*(1 - alpha)).astype(np.uint8) # remove transparency with a white background
		frames.append(frame)
	# make the last frame twice as long
	frames.append(frames[-1])
	# save it all as a GIF
	mimsave(f"results/animation-{num_frames}.gif", frames, fps=frame_rate)
	print(f"saved 'results/animation-{num_frames}.gif'!")


if __name__ == "__main__":
	main()
