import numpy as np
from src.shape_utility import np1, np3to2
from numpy.testing import assert_almost_equal


def test_np1():
    np1(np.array([[1, 2, 3], [4, 5, 6]]))
    assert_almost_equal(
        np1(np.array([[1, 2, 3], [4, 5, 6]])),
        np.arange(1, 7)
    )


def test_np3to2():
    arr1 = np3to2(np.array([
                [[0, 0], [1, 1], [2, 2]],
                [[3, 3], [4, 4], [5, 5]],
                [[6, 6], [7, 7], [8, 8]],
                [[9, 9], [10, 10], [11, 11]]
            ]))
    arr2 = np.array([
                [0, 0, 1, 1, 2, 2],
                [3, 3, 4, 4, 5, 5],
                [6, 6, 7, 7, 8, 8],
                [9, 9, 10, 10, 11, 11]
            ])
    assert_almost_equal(arr1, arr2)
