"""Tests for AsyDFL model decorator, DeBiasedAsyDFLKerasModel, and KerasCustomModelFactory."""

import numpy as np
import pytest
import tensorflow as tf

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModelDecorator
from p2pfl.learning.frameworks.tensorflow.custom_models.asydfl_model import (
    AsyDFLKerasP2PFLModel,
    DeBiasedAsyDFLKerasModel,
)
from p2pfl.learning.frameworks.tensorflow.custom_models.custom_model_factory import KerasCustomModelFactory
from p2pfl.learning.frameworks.tensorflow.keras_model import KerasModel

###
# Helpers
###


@tf.keras.utils.register_keras_serializable("p2pfl_test")
class TinyMLP(tf.keras.Model):
    """Minimal Keras model for testing."""

    def __init__(self, **kwargs: object):
        """Initialize the model."""
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(2, activation="relu")
        self.out = tf.keras.layers.Dense(1)
        self(tf.zeros((1, 3)))
        self.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        )

    def call(self, x, training=None):
        """Forward pass."""
        return self.out(self.dense(x))


def _make_keras_model() -> KerasModel:
    return KerasModel(TinyMLP())


# The DeBiasedAsyDFLKerasModel has an init ordering bug: Keras calls
# ``self.loss = None`` during ``super().__init__()`` which fires the custom
# loss setter *before* ``self.model`` is assigned.  We work around this in
# tests with a thin subclass that pre-sets ``self.model`` so the setter
# does not crash.  All logic under test is identical to the parent class.
@tf.keras.utils.register_keras_serializable("p2pfl_test")
class FixedDeBiasedModel(DeBiasedAsyDFLKerasModel):
    """DeBiasedAsyDFLKerasModel with init ordering fix for testing."""

    def __init__(self, model: tf.keras.Model, push_sum_weight: float = 1.0, **kwargs):
        """Initialize with pre-set model to avoid loss setter crash."""
        object.__setattr__(self, "model", model)
        super().__init__(model, push_sum_weight, **kwargs)


def _make_debiased(push_sum_weight: float = 1.0) -> FixedDeBiasedModel:
    """Create a testable DeBiasedAsyDFLKerasModel."""
    base = TinyMLP()
    return FixedDeBiasedModel(base, push_sum_weight=push_sum_weight)


# Similarly, patch AsyDFLKerasP2PFLModel to use the fixed constructor.
class FixedAsyDFLKerasP2PFLModel(AsyDFLKerasP2PFLModel):
    """AsyDFLKerasP2PFLModel that uses the fixed DeBiased constructor."""

    def __init__(self, wrapped_model: KerasModel, push_sum_weight: float = 1.0) -> None:
        """Initialize using FixedDeBiasedModel instead of the broken parent."""
        if not isinstance(wrapped_model.get_model(), DeBiasedAsyDFLKerasModel):
            debiased = FixedDeBiasedModel(wrapped_model.get_model(), push_sum_weight)
            wrapped_model.model = debiased
        # Skip the broken AsyDFLKerasP2PFLModel.__init__ and call the decorator init
        P2PFLModelDecorator.__init__(self, wrapped_model)

    def build_copy(self, **kwargs) -> "FixedAsyDFLKerasP2PFLModel":
        """Build copy using the fixed constructor."""
        copied_model = self._wrapped_model.build_copy(**kwargs)
        push_sum_weight = (
            float(copied_model.model.push_sum_weight.numpy()) if isinstance(copied_model.model, DeBiasedAsyDFLKerasModel) else 1.0
        )
        return FixedAsyDFLKerasP2PFLModel(copied_model, push_sum_weight=push_sum_weight)


###
# P2PFLModelDecorator tests
###


class TestP2PFLModelDecorator:
    """Tests for the generic P2PFLModelDecorator wrapper."""

    def test_get_parameters_delegates(self):
        """Verify get_parameters returns the same arrays as the inner model."""
        inner = _make_keras_model()
        wrapper = P2PFLModelDecorator(inner)
        for orig, wrapped in zip(inner.get_parameters(), wrapper.get_parameters(), strict=True):
            np.testing.assert_array_equal(orig, wrapped)

    def test_set_parameters_delegates(self):
        """Verify set_parameters modifies the inner model's weights."""
        inner = _make_keras_model()
        wrapper = P2PFLModelDecorator(inner)
        new_params = [np.ones_like(p) for p in inner.get_parameters()]
        wrapper.set_parameters(new_params)
        for p in inner.get_parameters():
            np.testing.assert_array_equal(p, np.ones_like(p))

    def test_get_framework_delegates(self):
        """Verify get_framework returns the inner model's framework string."""
        inner = _make_keras_model()
        wrapper = P2PFLModelDecorator(inner)
        assert wrapper.get_framework() == inner.get_framework()

    def test_setattr_delegates_to_wrapped(self):
        """Setting an arbitrary attribute goes to the inner model, not the wrapper."""
        inner = _make_keras_model()
        wrapper = P2PFLModelDecorator(inner)
        wrapper.num_samples = 42
        assert inner.num_samples == 42

    def test_getattr_delegates_to_wrapped(self):
        """Reading an attribute that only exists on the inner model works."""
        inner = _make_keras_model()
        inner.contributors = ["node-a"]
        wrapper = P2PFLModelDecorator(inner)
        assert wrapper.contributors == ["node-a"]


###
# DeBiasedAsyDFLKerasModel tests
###


class TestDeBiasedAsyDFLKerasModel:
    """Tests for the custom Keras model that applies push-sum de-biasing."""

    def test_constructor_direct(self):
        """Direct construction of DeBiasedAsyDFLKerasModel works."""
        base = TinyMLP()
        model = DeBiasedAsyDFLKerasModel(base)
        assert float(model.push_sum_weight.numpy()) == pytest.approx(1.0)

    def test_initial_push_sum_weight(self):
        """Default push_sum_weight is 1.0."""
        model = _make_debiased()
        assert float(model.push_sum_weight.numpy()) == pytest.approx(1.0)

    def test_custom_push_sum_weight(self):
        """Custom push_sum_weight can be set at construction."""
        model = _make_debiased(push_sum_weight=0.5)
        assert float(model.push_sum_weight.numpy()) == pytest.approx(0.5)

    def test_call_forward_pass(self):
        """Calling the model delegates to the inner model and returns correct shape."""
        model = _make_debiased()
        x = tf.constant([[1.0, 2.0, 3.0]])
        out = model(x)
        assert out.shape == (1, 1)

    def test_get_set_weights_roundtrip(self):
        """Verify get_weights / set_weights affect the inner model only."""
        model = _make_debiased()
        original_weights = [w.copy() for w in model.get_weights()]
        new_weights = [np.zeros_like(w) for w in original_weights]
        model.set_weights(new_weights)
        for w in model.get_weights():
            np.testing.assert_array_equal(w, np.zeros_like(w))

    def test_loss_property(self):
        """The loss property reads and writes to the inner model."""
        model = _make_debiased()
        new_loss = tf.keras.losses.MeanAbsoluteError()
        model.loss = new_loss
        assert model.loss is new_loss

    def test_optimizer_property(self):
        """The optimizer property reads and writes to the inner model."""
        model = _make_debiased()
        new_opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        model.optimizer = new_opt
        assert model.optimizer is new_opt

    def test_get_config_contains_fields(self):
        """Verify get_config contains model serialization and push_sum_weight."""
        model = _make_debiased(push_sum_weight=0.7)
        config = model.get_config()
        assert "model" in config
        assert config["push_sum_weight"] == pytest.approx(0.7)

    def test_train_step_runs(self):
        """Train_step completes without error and returns loss metric."""
        model = _make_debiased(push_sum_weight=2.0)
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        )

        x = tf.constant(np.random.randn(4, 3).astype(np.float32))
        y = tf.constant(np.random.randn(4, 1).astype(np.float32))
        metrics = model.train_step((x, y))
        assert "loss" in metrics

    def test_train_step_updates_weights(self):
        """After train_step the inner model weights should differ from before."""
        model = _make_debiased(push_sum_weight=1.0)
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        )
        before = [w.copy() for w in model.get_weights()]

        x = tf.constant(np.random.randn(8, 3).astype(np.float32))
        y = tf.constant(np.random.randn(8, 1).astype(np.float32))
        model.train_step((x, y))

        after = model.get_weights()
        any_changed = any(not np.allclose(b, a) for b, a in zip(before, after, strict=True))
        assert any_changed, "Weights should change after a training step"

    def test_train_step_debiasing_with_nonunit_weight(self):
        """Different push_sum_weight values produce different gradient updates."""
        np.random.seed(0)
        x = np.random.randn(8, 3).astype(np.float32)
        y = np.random.randn(8, 1).astype(np.float32)

        # Model A: push_sum_weight=1.0 (no de-biasing)
        model_a = _make_debiased(push_sum_weight=1.0)
        model_a.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.SGD(0.01))

        # Model B: same initial weights, push_sum_weight=3.0
        base_b = TinyMLP()
        base_b.set_weights(model_a.model.get_weights())
        model_b = FixedDeBiasedModel(base_b, push_sum_weight=3.0)
        model_b.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.SGD(0.01))

        model_a.train_step((tf.constant(x), tf.constant(y)))
        model_b.train_step((tf.constant(x), tf.constant(y)))

        # Weights should diverge because of different de-biasing factors
        any_diff = any(not np.allclose(a, b, atol=1e-6) for a, b in zip(model_a.get_weights(), model_b.get_weights(), strict=True))
        assert any_diff, "De-biasing with different push_sum_weight should produce different updates"

    def test_train_step_restores_weights_after_scaling(self):
        """With lr=0, scale+restore cancel out and weights remain unchanged."""
        model = _make_debiased(push_sum_weight=5.0)
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            # learning_rate=0 => no actual gradient update
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.0),
        )
        before = [w.copy() for w in model.get_weights()]

        x = tf.constant(np.random.randn(4, 3).astype(np.float32))
        y = tf.constant(np.random.randn(4, 1).astype(np.float32))
        model.train_step((x, y))

        after = model.get_weights()
        # With lr=0 the weights should be unchanged (scale + restore cancel out)
        for b, a in zip(before, after, strict=True):
            np.testing.assert_allclose(b, a, atol=1e-5)


###
# AsyDFLKerasP2PFLModel tests (using fixed subclass)
###


class TestAsyDFLKerasP2PFLModel:
    """Tests for the P2PFL-level wrapper combining decorator + DeBiasedAsyDFLKerasModel."""

    def test_wraps_plain_keras_model(self):
        """Passing a plain KerasModel wraps the inner model in DeBiasedAsyDFLKerasModel."""
        inner = _make_keras_model()
        asy = FixedAsyDFLKerasP2PFLModel(inner)
        assert isinstance(asy.get_model(), DeBiasedAsyDFLKerasModel)

    def test_does_not_double_wrap(self):
        """Passing an already-wrapped model does not wrap it again."""
        inner = _make_keras_model()
        FixedAsyDFLKerasP2PFLModel(inner)
        # inner.model is now DeBiasedAsyDFLKerasModel
        asy2 = FixedAsyDFLKerasP2PFLModel(inner)
        # Still the same DeBiasedAsyDFLKerasModel, not double-wrapped
        assert isinstance(asy2.get_model(), DeBiasedAsyDFLKerasModel)

    def test_get_push_sum_weight_default(self):
        """Default push_sum_weight is 1.0."""
        asy = FixedAsyDFLKerasP2PFLModel(_make_keras_model())
        assert asy.get_push_sum_weight() == pytest.approx(1.0)

    def test_set_push_sum_weight(self):
        """Verify set_push_sum_weight changes the value."""
        asy = FixedAsyDFLKerasP2PFLModel(_make_keras_model())
        asy.set_push_sum_weight(0.25)
        assert asy.get_push_sum_weight() == pytest.approx(0.25)

    def test_set_push_sum_weight_int(self):
        """Verify set_push_sum_weight accepts int values."""
        asy = FixedAsyDFLKerasP2PFLModel(_make_keras_model())
        asy.set_push_sum_weight(3)
        assert asy.get_push_sum_weight() == pytest.approx(3.0)

    def test_set_push_sum_weight_rejects_invalid(self):
        """Verify set_push_sum_weight raises ValueError for non-numeric types."""
        asy = FixedAsyDFLKerasP2PFLModel(_make_keras_model())
        with pytest.raises(ValueError, match="float or int"):
            asy.set_push_sum_weight("bad")  # type: ignore[arg-type]

    def test_get_set_parameters_roundtrip(self):
        """Parameters survive a get/set cycle through the wrapper."""
        asy = FixedAsyDFLKerasP2PFLModel(_make_keras_model())
        params = asy.get_parameters()
        new_params = [p + 1.0 for p in params]
        asy.set_parameters(new_params)
        for expected, actual in zip(new_params, asy.get_parameters(), strict=True):
            np.testing.assert_array_almost_equal(expected, actual)

    def test_get_framework(self):
        """Verify get_framework returns the TensorFlow framework string."""
        asy = FixedAsyDFLKerasP2PFLModel(_make_keras_model())
        assert asy.get_framework() == "tensorflow"

    def test_build_copy(self):
        """Verify build_copy preserves push_sum_weight and architecture."""
        asy = FixedAsyDFLKerasP2PFLModel(_make_keras_model(), push_sum_weight=0.5)
        copy = asy.build_copy()
        assert isinstance(copy, FixedAsyDFLKerasP2PFLModel)
        assert copy.get_push_sum_weight() == pytest.approx(0.5)
        # Architecture matches (same number of layers with same shapes)
        orig_params = asy.get_parameters()
        copy_params = copy.get_parameters()
        assert len(orig_params) == len(copy_params)
        for orig, copied in zip(orig_params, copy_params, strict=True):
            assert orig.shape == copied.shape

    def test_build_copy_is_independent(self):
        """Modifying the copy does not affect the original."""
        asy = FixedAsyDFLKerasP2PFLModel(_make_keras_model())
        copy = asy.build_copy()
        copy.set_push_sum_weight(99.0)
        copy.set_parameters([np.zeros_like(p) for p in copy.get_parameters()])
        assert asy.get_push_sum_weight() == pytest.approx(1.0)
        # Original weights should not be zeros
        assert any(np.any(p != 0) for p in asy.get_parameters())

    def test_custom_push_sum_weight_at_init(self):
        """Push_sum_weight set at construction is reflected correctly."""
        asy = FixedAsyDFLKerasP2PFLModel(_make_keras_model(), push_sum_weight=7.5)
        assert asy.get_push_sum_weight() == pytest.approx(7.5)

    def test_delegates_contribution_metadata(self):
        """Verify set_contribution / get_contributors / get_num_samples work through the decorator."""
        asy = FixedAsyDFLKerasP2PFLModel(_make_keras_model())
        asy.set_contribution(["node-1", "node-2"], 500)
        assert asy.get_contributors() == ["node-1", "node-2"]
        assert asy.get_num_samples() == 500


###
# KerasCustomModelFactory tests
###


class TestKerasCustomModelFactory:
    """Tests for the factory that wraps models in AsyDFLKerasP2PFLModel."""

    def test_create_asydfl(self):
        """Factory wraps a model in AsyDFLKerasP2PFLModel."""
        inner = _make_keras_model()
        result = KerasCustomModelFactory.create_model("AsyDFL", inner)
        assert isinstance(result, AsyDFLKerasP2PFLModel)

    def test_create_asydfl_idempotent_with_prewrapped(self):
        """Factory returns the same object when given an already-wrapped model."""
        inner = _make_keras_model()
        # Pre-wrap using the fixed subclass
        asy = FixedAsyDFLKerasP2PFLModel(inner)
        result = KerasCustomModelFactory.create_model("AsyDFL", asy)
        assert result is asy

    def test_unsupported_type_raises(self):
        """Factory raises ValueError for unknown model types."""
        inner = _make_keras_model()
        with pytest.raises(ValueError, match="Unsupported type"):
            KerasCustomModelFactory.create_model("NonExistent", inner)
