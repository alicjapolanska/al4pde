from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='al4pde',
    version='0.0.1',
    packages=find_packages(),
)
