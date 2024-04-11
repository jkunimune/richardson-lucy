"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>.
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal


def main():
	np.random.seed(0)

	x_kernel = np.linspace(0, 6, 201)
	Δx = x_kernel[1] - x_kernel[0]
	y_kernel = shoe_curve(x_kernel, 0.5, 1.0, +0.8, 1)

	x_source = np.linspace(0, 3, 101)
	y_source = 1000*(
			bell_curve(x_source, 0.3, 0.6, 4) +
			bell_curve(x_source, 0.7, 0.8, 2) +
			shoe_curve(x_source, 2.4, 1.2, -0.6, 3)
	)

	x_image = np.concatenate([x_kernel[:-1], x_kernel[-1] + x_source])

	# define the point-spread function as a matrix
	source_to_image = np.transpose(
		[signal.convolve(y, y_kernel, mode="full")*Δx for y in np.eye(x_source.size)]
	)
	source_to_data = (source_to_image[0:-1] + source_to_image[1:])/2
	source_to_image = source_to_image/np.sum(source_to_data, axis=0, keepdims=True)
	source_to_data = source_to_data/np.sum(source_to_data, axis=0, keepdims=True)
	y_data = np.random.poisson(source_to_data @ y_source)

	fig, (image_ax, kernel_ax, source_ax) = plt.subplots(3, 1, facecolor="none")

	image_ax.fill_between(np.repeat(x_image, 2)[1:-1], 0, np.repeat(y_data, 2))
	image_ax.set_xlim(0, 5)
	image_ax.set_ylim(0, None)

	kernel_ax.fill_between(x_kernel, 0, y_kernel)
	kernel_ax.set_xlim(0, 5)
	kernel_ax.set_ylim(0, None)

	source_ax.fill_between(x_source, 0, y_source)
	source_ax.set_xlim(0, 5)
	source_ax.set_ylim(0, None)

	plt.tight_layout()

	# then do the animation part
	source_fit, = source_ax.plot(x_source, np.zeros_like(x_source), color="C1")
	image_fit, = image_ax.plot(x_image, np.zeros_like(x_image), color="C1")
	label = image_ax.text(0.95, 0.90, "",
	                      horizontalalignment="right",
	                      verticalalignment="top",
	                      transform=image_ax.transAxes)
	y_source_guess = np.full(x_source.shape, np.sum(y_data)/np.sum(y_kernel)/(x_source[-1] - x_source[0]))
	for i in range(100):
		# refresh the values dependent on the current source guess
		y_image_guess = source_to_image @ y_source_guess
		y_data_guess = source_to_data @ y_source_guess

		# update the plot
		source_fit.set_ydata(y_source_guess)
		image_fit.set_ydata(y_image_guess)
		if i == 0:
			label.set_text(f"Initial guess")
		elif i == 1:
			label.set_text("1 iteration")
		else:
			label.set_text(f"{i} iterations")

		# Richardson–Lucy step
		y_source_guess = y_source_guess * (source_to_data.T @ (y_data/y_data_guess))
		plt.pause(.1)

	plt.show()


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
