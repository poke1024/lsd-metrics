# Author: Bernhard Liebl
# License: MIT

import numpy as np

from functools import lru_cache
from fractions import Fraction
from collections import namedtuple


LongShelhamerDarrellMetrics = namedtuple(
	'LongShelhamerDarrellMetrics',
	['pixel_accuracy', 'mean_accuracy', 'mean_IU', 'frequency_weighted_IU'])


def lsd_metrics(
	prediction: np.ndarray,
	truth: np.ndarray,
	n_classes: int) -> LongShelhamerDarrellMetrics:

	"""This computes the evaluation metrics given for semantic segmentation given in:
	[1] J. Long, E. Shelhamer, and T. Darrell, "Fully Convolutional Networks for
	Semantic Segmentation", 2014. (available at https://arxiv.org/abs/1411.4038).

	Note:
		Modified to exclude empty classes.

	Args:
		prediction: integer array of predicted classes for each pixel.
		truth: integer array of ground truth for each pixel.
		n_classes: defines the pixel classes [0, 1, ..., n_classes - 1].

	Returns:
		LongShelhamerDarrellMetrics: The computed metrics.
	"""

	def _check_array(name, a):
		if not np.issubdtype(a.dtype, np.integer):
			raise ValueError("given %s-array must be of type integer" % name)

		if not (0 <= np.min(a) < n_classes and 0 <= np.max(a) < n_classes):
			raise ValueError("non-class values in given %s-array" % name)

	_check_array('prediction', prediction)
	_check_array('truth', truth)

	classes = list(range(n_classes))

	@lru_cache(maxsize=None)
	def n(i: int, j: int) -> Fraction:
		# n(i, j) is "the number of pixels of class i predicted to belong to
		# class j", see [1].
		return Fraction(int(np.sum(np.logical_and(
			truth == i, prediction == j).astype(np.uint8), dtype=np.uint64)))

	@lru_cache(maxsize=None)
	def t(i: int) -> Fraction:
		# t(i) is "the total number of pixels of class i", see [1].
		return sum(n(j, i) for j in classes)

	non_empty_classes = [i for i in classes if t(i) > 0]

	return LongShelhamerDarrellMetrics(
		pixel_accuracy=sum(n(i, i) for i in classes) / sum(t(i) for i in classes),

		mean_accuracy=(Fraction(1) / len(non_empty_classes)) * sum(
			(n(i, i) / t(i)) for i in non_empty_classes),

		mean_IU=(Fraction(1) / len(non_empty_classes)) * sum(
			(
				n(i, i) / (
					t(i) + sum(n(j, i) for j in non_empty_classes) - n(i, i))
			) for i in non_empty_classes),

		frequency_weighted_IU=(Fraction(1) / sum(t(k) for k in non_empty_classes)) * sum(
			(
				(t(i) * n(i, i)) / (
					t(i) + sum(n(j, i) for j in non_empty_classes) - n(i, i))
			) for i in non_empty_classes)
	)


if __name__ == "__main__":
	# a simple demonstration.

	example_truth = np.array([
		[0, 0],
		[1, 1]
	])

	example_prediction = np.array([
		[0, 1],
		[1, 1]
	])

	print(lsd_metrics(example_prediction, example_truth, 2))
