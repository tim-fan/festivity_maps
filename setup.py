"""
Setup script for festivity-map package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="festivity-map",
    version="1.0.0",
    author="Tim Fan",
    description="A tool for mapping festive decorations using DINOv3 features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "Pillow",
        "torch",
        "torchvision",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "requests",
        "PyYAML",
        "folium",
    ],
    extras_require={
        "fiftyone": ["fiftyone"],
        "dev": ["pytest", "black", "flake8"],
    },
    entry_points={
        "console_scripts": [
            "festivity=festivity.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
