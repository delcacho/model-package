from setuptools import setup, find_packages

setup(
    name='model_package',
    version='0.6.2',
    packages=find_packages(),
    install_requires=[
        "mlflow==2.15.1", "matplotlib", "scikit-learn", "pycm"
    ],
)
