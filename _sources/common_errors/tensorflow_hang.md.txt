# TensorFlow Data Pipeline Hangs When Used with Ray or OpenDP

## Problem

When using TensorFlow's `tf.data.Dataset` (via HuggingFace's `to_tf_dataset()`) with p2pfl, the program hangs at:

```python
sample = next(iter(tf_dataset))
```

## Root Cause

**Import order conflict**: Ray or OpenDP is initialized before TensorFlow is imported, causing a threading deadlock.

1. Importing `p2pfl.management.logger` triggers `ray.init()` at module level
2. TensorFlow is imported afterwards
3. TensorFlow's data pipeline threads deadlock due to Ray's modified threading environment

```
p2pfl/management/logger/__init__.py:30 -> ray_installed() -> ray.init()
```

**This fails (hangs):**

```python
from p2pfl.management.logger import logger  # Ray initialized here
import tensorflow as tf  # Too late
from datasets import Dataset

dataset = Dataset.from_dict({"x": [[1]*784], "y": [0]})
tf_dataset = dataset.to_tf_dataset(batch_size=1, columns=["x"], label_cols=["y"])
next(iter(tf_dataset))  # Hangs forever
```

**This works:**

```python
import tensorflow as tf  # TensorFlow first
from p2pfl.management.logger import logger  # Ray after
from datasets import Dataset

dataset = Dataset.from_dict({"x": [[1]*784], "y": [0]})
tf_dataset = dataset.to_tf_dataset(batch_size=1, columns=["x"], label_cols=["y"])
next(iter(tf_dataset))  # Works
```

## Solutions

**Option 1: Import TensorFlow first (Quick Fix)**

Import TensorFlow before Ray or OpenDP:

```python
import tensorflow as tf  # FIRST
from p2pfl.management.logger import logger  # After TensorFlow
```

This is how p2pfl's test suite handles it in `test/conftest.py`:

```python
with contextlib.suppress(ImportError):
    import tensorflow
```

**Option 2: Don't install Ray**

Ray is an optional dependency. If you don't need distributed computing features:

```bash
pip install "p2pfl[tensorflow]"  # Without Ray
```

**Option 3: Disable Ray at runtime**

```python
from p2pfl.settings import Settings
Settings.general.DISABLE_RAY = True
```

## Environment

- TensorFlow 2.20.0
- Ray 2.53.0
- Python 3.12
- macOS (Darwin)

## Status

**Fixed on macOS** - p2pfl now uses a Ray worker setup hook to import TensorFlow before Ray workers start.

The fix is in `p2pfl/utils/check_ray.py`:

```python
def _worker_setup() -> None:
    """Import ML frameworks first in Ray workers to avoid deadlocks on macOS."""
    if sys.platform != "darwin":
        return
    import contextlib
    with contextlib.suppress(ImportError):
        import tensorflow
    with contextlib.suppress(ImportError):
        import torch

# In ray.init():
if sys.platform == "darwin":
    init_kwargs["runtime_env"] = {"worker_process_setup_hook": _worker_setup}
```

Related: [ray-project/ray#59661](https://github.com/ray-project/ray/issues/59661)
