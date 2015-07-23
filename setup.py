from setuptools import setup, find_packages

requirements = [req.strip() for req in open("requirements.txt").readlines()]

setup(
    name="nupic.vision",
    description="Vision tools and experiments for NuPIC",
    namespace_packages=["nupic"],
    packages = find_packages(),
    install_requires=requirements,
)
