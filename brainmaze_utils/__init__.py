"""
=======================================
Brainmaze utils (:mod:`brainmaze_utils`)
=======================================

"""
try:
    from setuptools_scm import get_version
    __version__ = get_version()
except LookupError:
    __version__ = 'dev'  # Fallback version, adjust as appropriate


# from brainmaze_utils import files
# from brainmaze_utils import signal
# from brainmaze_utils import types
# from brainmaze_utils import vector
#
# __all__ = ['files', 'signal', 'types', 'vector']
