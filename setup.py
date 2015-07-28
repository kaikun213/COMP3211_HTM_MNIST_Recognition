from setuptools import setup, find_packages

requirements = [req.strip() for req in open("requirements.txt").readlines()]

description = """Vision tools and experiments for NuPIC.

The ImageSensor was originally part of the main NuPIC repository and
has been moved here with other image and vision components and
experiments to reduce the dependencies for NuPIC users that aren't
doing vision-related work.
"""

setup(
    name="nupic.vision",
    description=description,
    namespace_packages=["nupic"],
    packages = find_packages(),
    install_requires=requirements,
)
