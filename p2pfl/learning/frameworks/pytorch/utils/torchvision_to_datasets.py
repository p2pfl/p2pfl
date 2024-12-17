"""
Example script that allows you to upload a PyTorch Vision dataset to the Hugging Face Hub.

The idea is to directly use parquet and datasets to reduce the overhead of creating a TorchVision dataset and then
transforming it to a Hugging Face dataset.

Usage:
```bash
poetry run python p2pfl/learning/pytorch/utils/torchvision_to_datasets.py \
	--dataset_name XXXX \
	--cache_dir ./data \
	--token XXXXXXXX \
	--dataset_summary "AAAA." \
	--dataset_description "XX" \
	--official_link "XXX" \
    --public
```

"""

import argparse
import os
import shutil
from typing import Any

import huggingface_hub  # type: ignore
from datasets import Dataset, DatasetDict  # type: ignore
from huggingface_hub import DatasetCard, DatasetCardData
from torchvision import datasets

SUPPORTED_DATASETS = [
    "CIFAR10",
    "CIFAR100",
    "MNIST",
    "FashionMNIST",
    "EMNIST",
    "QMNIST",
]


def create_huggingface_dataset_from_torchvision(
    dataset_name: str,
    cache_dir: str,
    train: bool = True,
    download: bool = True,
    generator_fn: Any = None,
) -> Dataset:
    """
    Create a Hugging Face dataset from a PyTorch Vision dataset.

    Args:
        dataset_name: The name of the dataset (e.g., "CIFAR10", "MNIST", "ImageNet").
        cache_dir: The directory where the dataset will be stored.
        train: Whether to load the training or test set.
        download: Whether to download the dataset if it's not found in the `root` directory.
        generator_fn: Optional custom generator function for specific dataset handling.

    Return:
        A Hugging Face dataset.

    """
    # Get dataset
    print(f"Creating dataset: {dataset_name} - Split: {'train' if train else 'test'}")
    dataset_class = getattr(datasets, dataset_name)
    if dataset_class is None:
        raise ValueError(f"Invalid dataset name: {dataset_name}. Available datasets: {datasets.__all__}")
    dataset = dataset_class(root=cache_dir, train=train, download=download)

    # Transform to a generic dataset (using generators)
    print("Transforming dataset.")

    def generate_examples():
        for image, label in dataset:  # Unpack image and label from the tuple
            yield {"image": image, "label": label}  # Yield a dictionary

    if generator_fn is not None:
        generate_examples = generator_fn(dataset)

    return Dataset.from_generator(generate_examples)


def __get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a PyTorch Vision dataset to the Hugging Face Hub.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The name of the dataset (e.g., 'CIFAR10', 'MNIST', 'ImageNet').",
        required=True,
        choices=SUPPORTED_DATASETS,
    )
    parser.add_argument("--cache_dir", type=str, help="The directory where the dataset will be stored.", required=True)
    parser.add_argument("--token", type=str, help="The Hugging Face API token.", required=True)
    parser.add_argument(
        "--no_remove_cache",
        action="store_true",
        help="Whether to remove the cache directory after uploading the dataset.",
    )
    parser.add_argument(
        "--dataset_summary",
        type=str,
        help="The summary of the dataset.",
    )
    parser.add_argument(
        "--dataset_description",
        type=str,
        help="The description of the dataset.",
    )
    parser.add_argument(
        "--license",
        type=str,
        help="The license of the dataset.",
        required=True,
    )
    parser.add_argument(
        "--official_link",
        type=str,
        help="The creators of the annotations.",
        required=True,
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Whether to make the dataset public.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Get args
    args = __get_args()

    # Check if the dataset is supported
    if args.dataset_name not in SUPPORTED_DATASETS:
        print(
            f"Warning: The dataset '{args.dataset_name}' is not officially supported. "
            "The license might not allow redistribution. "
            "You can easily modify the script to support other datasets, but please ensure you have the necessary permissions."
        )

    # Login
    huggingface_hub.login(args.token)

    # Create datasets for different splits (e.g., train, test, validation)
    train_dataset = create_huggingface_dataset_from_torchvision(args.dataset_name, args.cache_dir, train=True)
    test_dataset = create_huggingface_dataset_from_torchvision(args.dataset_name, args.cache_dir, train=False)

    # Combine datasets into a DatasetDict
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Push to Hub
    print("Pushing dataset to the Hugging Face Hub.")
    dataset.push_to_hub(repo_id=args.dataset_name, private=not args.public)

    # Get full repo name
    full_repo_name = huggingface_hub.get_full_repo_name(args.dataset_name)

    # Get template path
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(cur_dir, "dataset_card_template.md")
    print(template_path)

    # Create DatasetCardData
    print("Creating dataset card.")
    card_data = DatasetCard.load(full_repo_name).data.to_dict()
    card_data["language"] = "en"
    card_data["license"] = args.license
    card_data["task_categories"] = ["image-classification"]
    card_data["task_ids"] = ["multi-class-image-classification"]
    card_data["multilinguality"] = "monolingual"
    card_data["pretty_name"] = args.dataset_name

    link = f"[{args.dataset_name}]({args.official_link})"
    card = DatasetCard.from_template(
        card_data=DatasetCardData(**card_data),
        template_path=template_path,
        dataset_summary=args.dataset_summary,
        dataset_description=args.dataset_description,
        dataset_name=args.dataset_name,
        source_data=f"Auto-generated from PyTorch Vision, please check the original {link} for more info.",
    )

    print(f"Pushing dataset card to the Hugging Face Hub: {full_repo_name}")
    card.push_to_hub(full_repo_name, repo_type="dataset")

    print("🐞 Done.")

    # Remove cache
    if not args.no_remove_cache:
        print("Removing cache directory.")
        shutil.rmtree(args.cache_dir)
