from setuptools import find_packages, setup
from pathlib import Path

README = Path.joinpath("README.md").read_text()

setup(
    name='p2pfl', 
    version='0.1.0',
    author='Pedro Guijas',
    author_email="pguijas@gmail.com",
    description='p2p Federated Learning framework', 
    long_description=README,
    packages=find_packages(include=['p2pfl']), 
    install_requires=[], 
    setup_requires=[], 
    tests_require=[], 
    test_suite='test',
    #install_requires=REQUIREMENTS,
    #license='MIT',
)


