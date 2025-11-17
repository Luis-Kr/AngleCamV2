from setuptools import setup, find_packages

# Dependencies from environment.yml
install_deps = [
    "torch>=2.1.0",
    "torchvision>=0.16.0",
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.1.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "pillow>=10.0.0",
    "matplotlib>=3.7.0",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.0",
    "albumentations>=1.3.0",
    "tqdm>=4.65.0",
    "open3d>=0.18.0",
    "laspy>=2.5.0",
    "numba>=0.58.0",
    "opencv-python>=4.8.0",
]

setup(
    name="anglecam",
    version="2.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=install_deps,
)
