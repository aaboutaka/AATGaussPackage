import pathlib
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='AATGaussPackage',
    version='0.0.2',
    description='AAT Gaussian utility Package',
    long_description = long_description,
    long_description_content_type= 'text/markdown',
    url='https://github.com/aaboutaka/AATGaussPackage',
    author='Ali Abou Taka',
    author_email='abotaka.ali@gmail.com',
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        ],
    packages=['aatgausspackage'],
    include_package_data=True,
    zip_safe=False)
