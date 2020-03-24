import setuptools

setuptools.setup(
	name="cmbpix", 
	version="0.1.0.dev0", 
	author="Victor C. Chan", 
	author_email="chan@astro.utoronto.ca", 
	description="Tools for pixel-based CMB analysis", 
	packages=["cmbpix"], 
	python_requires=">=3", 
	install_requires=["numpy", "healpy"], 
)