# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest



def test_import():
    print("Testing import 'from brainmaze_utils.types'")
    try :
        import brainmaze_utils.types
        assert True
    except ImportError:
        assert False


# basedir = os.path.abspath(os.path.dirname(__file__))
#
# class TestVector(TestCase):
#     def test_import(self):
#         print("Testing import 'from brainmaze_utils.types'")
#
#
#
# if __name__ == '__main__':
#     unittest.main()
