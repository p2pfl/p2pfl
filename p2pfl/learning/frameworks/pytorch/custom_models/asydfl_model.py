"""Custom PyTorch Lightning model with de-biased gradient updates for AsyncDFL."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel, P2PFLModelDecorator
from p2pfl.learning.frameworks.pytorch.lightning_learner import resolve_optimizer


class DeBiasedAsyDFLModule(L.LightningModule):
    """
    Wrapper that applies push-sum de-biasing around the inner model's training step.

    Before computing gradients, all parameters are divided by push_sum_weight
    (the push-sum mixing scalar). After backward, parameters are restored
    before the optimizer step.

    Args:
        model: The inner LightningModule.
        push_sum_weight: The push-sum weight for de-biasing.

    """

    def __init__(self, model: L.LightningModule, push_sum_weight: float = 1.0) -> None:
        """Initialize the de-biased module."""
        super().__init__()
        self.model = model
        self._push_sum_weight = push_sum_weight
        self._cached_optimizer: torch.optim.Optimizer | None = None

    def _get_optimizer(self) -> torch.optim.Optimizer:
        if self._cached_optimizer is None:
            self._cached_optimizer = resolve_optimizer(self.model)
        return self._cached_optimizer

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass delegates to inner model."""
        return self.model(*args, **kwargs)

    def _delegate_step(self, step_fn: Any, batch: Any, batch_idx: int) -> torch.Tensor:
        """Call inner model's step, redirecting self.log() to the wrapper."""
        original_log = self.model.log
        self.model.log = self.log  # type: ignore[assignment]
        try:
            return step_fn(batch, batch_idx)  # type: ignore[no-any-return]
        finally:
            self.model.log = original_log  # type: ignore[assignment]

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        """Delegate to inner model's training_step."""
        return self._delegate_step(self.model.training_step, batch, batch_idx)

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        """Delegate to inner model's test_step."""
        return self._delegate_step(self.model.test_step, batch, batch_idx)

    def configure_optimizers(self) -> Any:
        """Delegate to inner model's optimizer configuration."""
        return self.model.configure_optimizers()

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """Proxy to inner model to keep parameter keys consistent."""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        """Proxy to inner model to keep parameter keys consistent."""
        return self.model.load_state_dict(state_dict, *args, **kwargs)

    def train_on_batch(self, batch: Any) -> float:
        """
        De-biased single-batch training for AsyncDFL push-sum consensus.

        1. Scale parameters by 1/push_sum_weight
        2. Forward pass + loss computation
        3. Backward pass (gradients computed on de-biased weights)
        4. Restore parameters by push_sum_weight
        5. Optimizer step (applies gradients to restored weights)

        Args:
            batch: A single training batch.

        Returns:
            The scalar loss value.

        """
        opt = self._get_optimizer()
        self.model.train()

        with torch.no_grad():
            for p in self.model.parameters():
                p.data.div_(self._push_sum_weight)

        # Suppress self.log() in inner model — no Trainer context here
        original_log = self.model.log
        self.model.log = lambda *a, **kw: None  # type: ignore[assignment]
        try:
            loss: torch.Tensor = self.model.training_step(batch, 0)  # type: ignore[assignment]
        finally:
            self.model.log = original_log  # type: ignore[assignment]

        loss.backward()  # type: ignore[no-untyped-call]

        with torch.no_grad():
            for p in self.model.parameters():
                p.data.mul_(self._push_sum_weight)

        opt.step()
        opt.zero_grad()

        return loss.item()


class AsyDFLLightningModel(P2PFLModelDecorator):
    """
    PyTorch Lightning P2PFL Model with de-biased gradient updates.

    Wraps a LightningModel with a DeBiasedAsyDFLModule for push-sum
    consensus training.

    Args:
        wrapped_model: The LightningModel to wrap.
        push_sum_weight: The push-sum weight for de-biasing.

    """

    def __init__(
        self,
        wrapped_model: P2PFLModel,
        push_sum_weight: float = 1.0,
    ) -> None:
        """Initialize the model."""
        inner_model = wrapped_model.get_model()
        if isinstance(inner_model, DeBiasedAsyDFLModule):
            inner_model._push_sum_weight = push_sum_weight
        else:
            debiased_model = DeBiasedAsyDFLModule(inner_model, push_sum_weight)
            wrapped_model.model = debiased_model
        super().__init__(wrapped_model)
        self.add_info("push_sum_weight", float(push_sum_weight))

    def get_push_sum_weight(self) -> float:
        """Get the push-sum weight."""
        return self.get_model()._push_sum_weight

    def set_push_sum_weight(self, weight: float | int) -> None:
        """Set the push-sum weight on both the module and additional_info."""
        if not isinstance(weight, float | int):
            raise ValueError("Push sum weight must be a float or int.")
        w = float(weight)
        self.get_model()._push_sum_weight = w
        self.add_info("push_sum_weight", w)

    def build_copy(self, **kwargs: Any) -> AsyDFLLightningModel:
        """Build a copy preserving push_sum_weight."""
        copied_model = self._wrapped_model.build_copy(**kwargs)

        push_sum_weight: float = (
            copied_model.model._push_sum_weight if isinstance(copied_model.model, DeBiasedAsyDFLModule) else 1.0
        )

        return AsyDFLLightningModel(
            wrapped_model=copied_model,
            push_sum_weight=push_sum_weight,
        )
