# Segmentation Fault When Using XGBoost with PyTorch (macOS)

## Problem

**Import order conflict on macOS**: Importing `xgboost` before `torch` causes a segmentation fault when calling `torch.tensor()` on numpy arrays.

**This fails (segfaults):**

```python
import xgboost as xgb  # XGBoost first
import torch
import numpy as np

t = torch.tensor(np.random.randn(256, 784).astype(np.float32))  # Segfault
```

**This works:**

```python
import torch  # PyTorch first
import xgboost as xgb
import numpy as np

t = torch.tensor(np.random.randn(256, 784).astype(np.float32))  # Works
```

## Solutions

**Option 1: Import PyTorch first**

Ensure `torch` is imported before `xgboost` in your code:

```python
import torch  # FIRST
import xgboost as xgb  # After PyTorch
```

**Option 2: Use separate environments**

When working with multiple ML frameworks (PyTorch, TensorFlow, XGBoost, etc.) on macOS, consider using separate virtual environments for each framework to avoid import order conflicts.

## Environment

- macOS (Apple Silicon / arm64)
- PyTorch 2.9.1
- XGBoost 3.1.2
- Python 3.12

## Important Note

macOS users should be particularly careful when installing multiple ML frameworks in the same environment. Import order conflicts between frameworks can cause subtle and hard-to-debug issues like segmentation faults or deadlocks.

## Status

**Open Issue** - Upstream PyTorch bug.

Tracked at: [pytorch/pytorch#171323](https://github.com/pytorch/pytorch/issues/171323)
