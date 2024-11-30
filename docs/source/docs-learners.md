# 🧠 Learners

## Overview

**Learners** implement the learning process. Each Learner is associated to a `P2PFLModel` and a `P2PFLDataset`. Each learner updates its model with the data from its dataset. If a learner is not specified when creating a `Node`, it is assigned by default depending on which framework is used. Learners are also responsible of transmiting information from `P2PFLCallbacks` to the `P2PFLModels` and viceversa.

## Features
- **Model training**: Trains and updates the Node's model using the given dataset.
- **Callback management** : Shares information between callbacks and models.

---

## Available Learners

Currently the library has support for **PyTorch Lighting** , **Keras** and **Flax** learners.
By default, the correct learner is assigned automatically depending on the model you are using.
If you want to implement a custom learner, you can make your nodes use it by setting the *learner class* as a parameter during initialization:

```python
node = Node(
    ...
    learner = YourCustomLearnerClass
)
```
🌟 Ready? **You can view next**: > [Communication Protocols](docs-communication-protocol.md)

<div style="position: fixed; bottom: 10px; right: 10px; font-size: 0.9em; padding: 10px; border: 1px solid #ccc; border-radius: 5px;"> 🌟 You Can View Next: <a href="docs-communication-protocol.md">Communication Protocols</a> </div>