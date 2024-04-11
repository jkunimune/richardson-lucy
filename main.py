"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>.
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal


TRUTH_COLOR = "#ff9444"
FIT_COLOR = "#7b0031"


def main():
	np.random.seed(1)

	# define the source
	x_source = np.linspace(0, 3, 101)
	y_source = 500*(
			bell_curve(x_source, 0.3, 0.6, 4) +
			bell_curve(x_source, 0.7, 0.6, 2) +
			shoe_curve(x_source, 2.2, 0.6, -0.5, 3)
	)

	# define the point-spread function
	x_kernel = np.linspace(0, 6, 201)
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
	fig, (image_ax, kernel_ax, source_ax) = plt.subplots(3, 1, facecolor="none")
	for ax in image_ax, kernel_ax, source_ax:
		ax.set_xticks([])
		ax.set_yticks([])

	image_ax.fill_between(np.repeat(x_image, 2)[1:-1], 0, np.repeat(y_data, 2), color=TRUTH_COLOR)
	image_ax.set_xlim(0, 5)
	image_ax.set_ylim(0, None)
	image_ax.set_ylabel("Measurement")

	kernel_ax.fill_between(x_kernel, 0, y_kernel, color=TRUTH_COLOR)
	kernel_ax.set_xlim(0, 5)
	kernel_ax.set_ylim(0, None)
	kernel_ax.set_ylabel("Point-spread function")

	source_ax.fill_between(x_source, 0, y_source, color=TRUTH_COLOR)
	source_ax.set_xlim(0, 5)
	source_ax.set_ylim(0, None)
	source_ax.set_ylabel("Object")

	plt.tight_layout()

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
	indices = indices[:np.argmin(error[indices]) + 2]  # stop after the rms error increases once

	# then do the animation part
	source_fit, = source_ax.plot(x_source, np.zeros_like(x_source), linestyle="dashed", color=FIT_COLOR, linewidth=1.5)
	image_fit, = image_ax.plot(x_image, np.zeros_like(x_image), linestyle="dashed", color=FIT_COLOR, linewidth=1.5)
	label = image_ax.text(0.95, 0.90, "",
	                      horizontalalignment="right",
	                      verticalalignment="top",
	                      transform=image_ax.transAxes)
	for i in indices:
		y_image_guess = source_to_image @ y_source_guesses[i]
		source_fit.set_ydata(y_source_guesses[i])
		image_fit.set_ydata(y_image_guess)
		if i == 0:
			label.set_text(f"Initial guess")
		elif i == 1:
			label.set_text("1 iteration")
		else:
			label.set_text(f"{i} iterations")
		plt.pause(.8)

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


if __name__ == "__main__":
	main()
