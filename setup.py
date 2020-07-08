from setuptools import setup, find_packages
import os

setup(
    name="detr",
    packages=find_packages(),
    install_requires=["torch", "torchvision"],
    version="0.1"
)
