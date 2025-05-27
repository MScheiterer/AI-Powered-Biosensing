from setuptools import setup, find_packages

setup(
    name="AI-Powered-Biosensing",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)