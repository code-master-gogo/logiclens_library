from setuptools import setup, find_packages

setup(
    name="fastpersondetect",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5",
        "numpy>=1.19",
        "tensorflow>=2.10",
        "scikit-image>=0.19"
    ],
    author="Anubhav Pandey",
    description="A fast and accurate CPU-based person detection library",
    url="https://github.com/code-master-gogo/logiclens_library",
)