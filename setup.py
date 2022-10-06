# 
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/federated_learning_p2p).
# Copyright (c) 2022 Pedro Guijas Bravo.
# 
# This program is free software: you can redistribute it and/or modify  
# it under the terms of the GNU General Public License as published by  
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

from setuptools import find_packages, setup
from pathlib import Path

HERE = Path(__file__).parent

PACKAGE_NAME = "p2pfl"
VERSION = "0.1.7"
AUTHOR = "Pedro Guijas"
AUTHOR_EMAIL = "pguijas@gmail.com"
URL = "https://pguijas.github.io/federated_learning_p2p/"
DESCRIPTION = "p2p Federated Learning framework"
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
LONG_DESC_TYPE = "text/markdown"
LICENSE = "MIT"

INSTALL_REQUIRES = [
    "torch==1.11.0",
    "tensorboard",
    "pytorch-lightning",
    "torchvision",
    "pycryptodome",
]  # dependencias


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
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="test",
)
