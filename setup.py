# -*- coding: utf-8

from setuptools import setup, find_packages

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ramses2",
    version="1.0",
    author="Jerome Lux",
    description="Recycled Aggregates Mass Estimation and Segmentation (RAMSES) — SOLOv2-based instance segmentation with mass estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jerome-lux/RAMSES2",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: APACHEv2 License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=9.0.0",
        "scikit-image>=0.19.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "pytorch-lightning>=1.6.0",
        "tensorboard>=2.8.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
            "ipykernel>=6.0.0",
        ],
        "coco": [
            "pycocotools>=2.0.2",
        ],
        "gui": [
            "PyQt5>=5.15.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
