"""
Setup script for CNN-Mamba-SSL project.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cnn-mamba-ssl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Self-supervised contrastive learning for heart sound classification using CNN-Mamba hybrid encoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dunkmonkey/CNN-Mamba-SSL",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "lightning>=2.0.0",
        "hydra-core>=1.3.0",
        "mamba-ssm>=1.0.0",
        "librosa>=0.10.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
    },
)
