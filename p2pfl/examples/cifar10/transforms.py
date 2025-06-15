#
# This file is part of the federated_learning_p2p (p2pfl) distribution (see https://github.com/pguijas/p2pfl).
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

"""
CIFAR10 transforms for P2PFL datasets.

This module contains transformation functions for CIFAR10 datasets.
"""

import torchvision.transforms as transforms


def cifar10_train_transforms(examples):
    """Apply training transforms (with data augmentation) to a batch of CIFAR10 examples."""
    # Define training transforms with data augmentation
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    # Transform all images using list comprehension
    transformed_images = [train_transform(img) for img in examples["image"]]

    return {"image": transformed_images, "label": examples["label"]}


def cifar10_test_transforms(examples):
    """Apply test transforms (normalization only) to a batch of CIFAR10 examples."""
    # Define test transforms (no data augmentation)
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])]
    )

    # Transform all images using list comprehension
    transformed_images = [test_transform(img) for img in examples["image"]]

    return {"image": transformed_images, "label": examples["label"]}


def get_cifar10_transforms():
    """
    Get the CIFAR10 transforms.

    Returns a dictionary with separate transforms for train and test splits.
    This would allow different transforms for each split if the framework supported it.

    Note: The current P2PFLDataset implementation applies the same transform to all splits.
    To use different transforms for train/test, the framework's set_transforms method
    would need to be modified to handle dictionary returns.

    Returns:
        dict: A dictionary mapping split names to transform functions

    """
    return {"train": cifar10_train_transforms, "test": cifar10_test_transforms}
