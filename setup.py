import setuptools
from Cython.Build import cythonize
import numpy as np

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

extensions = [setuptools.Extension("cmbpix.lensing.SCALE_c", 
	["cmbpix/lensing/SCALE_c.pyx"], 
	extra_compile_args=['-fopenmp'], 
	extra_link_args=['-fopenmp']), 
]

setuptools.setup(
	name="cmbpix", 
	version="1.0.0", 
	author="Victor C. Chan", 
	author_email="chan@astro.utoronto.ca", 
	description="Tools for pixel-based CMB analysis", 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
	packages=setuptools.find_packages(include=["cmbpix", "cmbpix.lensing"]), 
	python_requires=">=3", 
	install_requires=["numpy", "healpy", "cython", "pixell", "matplotlib", 
                   		"mpi4py", "camb"], 
	ext_modules=cythonize(extensions, 
		compiler_directives={"language_level": "3"}), 
	include_dirs=[np.get_include()], 
)
