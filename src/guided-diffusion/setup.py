from setuptools import setup

setup(
    name="guided-diffusion",
    py_modules=["guided-diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
