import distutils.core
import Cython.Build
distutils.core.setup(ext_modules = Cython.Build.cythonize("xyz.pyx"))

"""
Then run: 
    python cython_setup.py build
"""
