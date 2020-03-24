import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
	name="cmbpix", 
	version="0.1.0.dev0", 
	author="Victor C. Chan", 
	author_email="chan@astro.utoronto.ca", 
	description="Tools for pixel-based CMB analysis", 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
	packages=setuptools.find_packages(include=["cmbpix", "cmbpix.lensing"]), 
	python_requires=">=3", 
	install_requires=["numpy", "healpy"], 
)