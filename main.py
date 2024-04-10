"""
This work by Justin Kunimune is marked with CC0 1.0 Universal.
To view a copy of this license, visit <https://creativecommons.org/publicdomain/zero/1.0>.
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy import signal


def main():
	x = np.linspace(0, 5, 2001)
	Δx = x[1] - x[0]

	y_kernel = shoe_curve(x, 0.5, 1.0, +0.8, 1000)

	y_source = bell_curve(x, 0.3, 0.6, 4) + bell_curve(x, 0.7, 0.8, 2) + shoe_curve(x, 2.4, 1.2, -0.6, 3)

	y_image = signal.convolve(y_source, y_kernel*Δx, mode="full")[:x.size]

	fig, (image_ax, kernel_ax, source_ax) = plt.subplots(3, 1, facecolor="none")

	image_ax.plot(x, y_image)
	image_ax.set_xlim(0, 5)
	image_ax.set_ylim(0, None)

	kernel_ax.plot(x, y_kernel)
	kernel_ax.set_xlim(0, 5)
	kernel_ax.set_ylim(0, None)

	source_ax.plot(x, y_source)
	source_ax.set_xlim(0, 5)
	source_ax.set_ylim(0, None)

	plt.tight_layout()
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