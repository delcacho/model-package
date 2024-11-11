from setuptools import setup, find_packages

setup(
    name='model_package',
    version='0.4',
    packages=find_packages(),
    install_requires=[
        "mlflow==2.17.2", "matplotlib", "scikit-learn", "pycm"
    ],
)
