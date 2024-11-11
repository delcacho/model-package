from setuptools import setup, find_packages

setup(
    name='model_package',
    version='0.6',
    packages=find_packages(),
    install_requires=[
        "mlflow", "matplotlib", "scikit-learn", "pycm"
    ],
)
