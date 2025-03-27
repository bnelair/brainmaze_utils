# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import numpy as np
from brainmaze_utils.vector import rotate, _check_scale, _check_dimensions, translate, scale


def test_import():
    print("Testing import 'from brainmaze_utils.vector'")
    try :
        import brainmaze_utils.vector
        assert True
    except ImportError:
        assert False


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_valid_dimensions(dim):
    x = np.random.rand(5, dim)
    assert _check_dimensions(x)

def test_invalid_dimensions_too_low():
    x = np.random.rand(5, 0)
    with pytest.raises(AssertionError):
        _check_dimensions(x)

def test_invalid_dimensions_too_high():
    x = np.random.rand(5, 4)
    with pytest.raises(AssertionError):
        _check_dimensions(x)

def test_valid_scale():
    x = np.random.rand(5, 3)
    m = [1, 2, 3]
    assert _check_scale(x, m)

def test_invalid_scale_length_mismatch():
    x = np.random.rand(5, 3)
    m = [1, 2]
    with pytest.raises(AssertionError):
        _check_scale(x, m)



# TODO: Test for this have to be implemented. Examples below are just not working suggestions
#
# def test_translate(data_vectors):
#     translated = translate(data_vectors['data_2d'].copy(), [1, 2])
#     expected = np.array([[2, 4], [4, 6]])
#     np.testing.assert_array_equal(translated, expected)
#
# def test_scale(data_vectors):
#     scaled = scale(data_vectors['data_2d'].copy(), [1, 2])
#     expected = np.array([[0., 0.], [4., 8.]])  # Adjust expected values based on your scale logic
#     np.testing.assert_allclose(scaled, expected)
#
# def test_rotate_2d(data_vectors):
#     rotated = rotate(data_vectors['data_2d'].copy(), 90)
#     expected = np.array([[-2, 1], [-4, 3]])  # Expected results for 90-degree rotation
#     np.testing.assert_allclose(rotated, expected, atol=1e-7)
#
# def test_rotate_3d(data_vectors):
#     rotated = rotate(data_vectors['data_3d'].copy(), [90, 90, 90])
#     expected = data_vectors['data_3d']  # Adjust expected values based on your 3D rotation logic
#     np.testing.assert_allclose(rotated, expected, atol=1e-7)



