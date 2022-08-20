from setuptools import find_packages, setup
from pathlib import Path

HERE = Path(__file__).parent

PACKAGE_NAME = 'p2pfl' 
VERSION = '0.1.6' 
AUTHOR = 'Pedro Guijas' 
AUTHOR_EMAIL = 'pguijas@gmail.com' 
URL = 'https://pguijas.github.io/federated_learning_p2p/' 
DESCRIPTION = 'p2p Federated Learning framework' 
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding='utf-8')
LONG_DESC_TYPE = "text/markdown"
LICENSE = 'MIT' 

INSTALL_REQUIRES = [
    "torch==1.11.0",
    "tensorboard",
    "pytorch-lightning",
    "torchvision",
    "pycryptodome"
] # dependencias


setup(
    name=PACKAGE_NAME, 
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    license=LICENSE,
    install_requires=INSTALL_REQUIRES,
    package_dir={"p2pfl": "p2pfl"},
    packages=find_packages(
        where=".",
        include=[
            "*",
        ],
        exclude=[
            "test",
        ],
    ),
    include_package_data=True,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='test'
)


