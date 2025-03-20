# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import numpy as np

from copy import deepcopy
from shutil import rmtree

import pytest


def test_import():
    print("Testing import 'from brainmaze_utils.files'")
    import brainmaze_utils.files
    try :
        import brainmaze_utils.files
        assert True
    except ImportError:
        assert False


#
# class TestFiles(TestCase):
#     def test_import(self):
#         print("Testing import 'from brainmaze_utils.files'")
#
#
# if __name__ == '__main__':
#     unittest.main()
