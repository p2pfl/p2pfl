![GitHub Logo](../../../other/logo.png)

# P2PFL - CIFAR-10 Example

This folder contains an example of using P2PFL with the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Implementation Details

- **Model**: LeNet-5 convolutional neural network
- **Dataset**: CIFAR-10 from Hugging Face
- **Frameworks**: PyTorch (with Lightning)

## How to Run

> Be sure that p2pfl is installed

### Running `cifar10.yaml`

To run the example using the YAML configuration:

```sh
python -m p2pfl run cifar10.yaml
```

You can modify the YAML file to customize the experiment parameters.
