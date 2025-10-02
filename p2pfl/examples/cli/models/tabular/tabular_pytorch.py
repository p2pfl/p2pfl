from typing import Dict, Optional, Sequence

import lightning as L
import torch
from torchmetrics import Accuracy, Metric, F1Score
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed


class MLP(L.LightningModule):
    """MLP for arbitrary tabular features."""

    def __init__(
            self,
            feature_keys: Sequence[str],
            out_channels: int,
            hidden_sizes: Optional[Sequence[int]] = None,
            activation: str = "relu",
            metric: type[Metric] = Accuracy,
            lr_rate: float = 1e-3,
            label_key: str = "label",
    ) -> None:
        """
        Args:
            feature_keys: list of keys in batch dict to use as inputs.
            out_channels: number of target classes (1 for binary).
            hidden_sizes: sizes of hidden layers.
            activation: activation fn name.
            metric: torchmetrics class.
            lr_rate: learning rate.
            label_key: key for target in batch dict.
        """
        super().__init__()
        set_seed(Settings.general.SEED, "pytorch")

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        self.feature_keys = list(feature_keys)
        input_size = len(self.feature_keys)

        # check out_channels
        self.out_channels = out_channels
        self.lr_rate = lr_rate

        # choose metric
        self.metric = Accuracy(task="binary" if self.out_channels == 1 else "multiclass", num_classes=self.out_channels if self.out_channels > 1 else None)
        self.f1 = F1Score(task="binary" if self.out_channels == 1 else "multiclass", num_classes=self.out_channels if self.out_channels > 1 else None)

        self.label_key = label_key

        # build network
        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
        layers.append(self._get_activation(activation))
        for i in range(len(hidden_sizes) - 1):
            layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(self._get_activation(activation))
        layers.append(torch.nn.Linear(hidden_sizes[-1], self.out_channels))

        self.layers = torch.nn.ModuleList(layers)

    def _get_activation(self, name: str) -> torch.nn.Module:
        if name == "relu":
            return torch.nn.ReLU()
        elif name == "sigmoid":
            return torch.nn.Sigmoid()
        elif name == "tanh":
            return torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass through MLP and return logits."""
        for layer in self.layers:
            x = layer(x)
        return x

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr_rate, weight_decay=1e-4)
        # return torch.optim.Adam(self.parameters(), lr=self.lr_rate, weight_decay=1e-4)

    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Concatena todas las feature_keys en un solo tensor [batch, features]."""
        cols = [batch[k].unsqueeze(1) for k in self.feature_keys]
        x = torch.cat(cols, dim=1).float()
        return x

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        x = self._prepare_batch(batch)
        y = batch[self.label_key]
        # Validar que las etiquetas estén en el rango correcto
        if not torch.all((y >= 0) & (y < self.out_channels)):
            raise ValueError(f"Las etiquetas en el batch no están en el rango [0, {self.out_channels-1}]: {y}")
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        x = self._prepare_batch(batch)
        y = batch[self.label_key]
        # Validar que las etiquetas estén en el rango correcto
        if not torch.all((y >= 0) & (y < self.out_channels)):
            raise ValueError(f"Las etiquetas en el batch no están en el rango [0, {self.out_channels-1}]: {y}")
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.metric(preds, y)
        f1 = self.f1(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)
        # Calcular y loguear precision y recall

        y_true = y.cpu().numpy()
        y_score = torch.softmax(logits, dim=1).cpu().detach().numpy()
        precision = precision_score(y_true, preds.cpu().numpy(), average='weighted', zero_division=0)
        recall = recall_score(y_true, preds.cpu().numpy(), average='weighted', zero_division=0)
        self.log("test_precision", precision, prog_bar=True)
        self.log("test_recall", recall, prog_bar=True)
        if self.out_channels > 1:
            # AUC para multiclase (one-vs-rest), robusto a clases ausentes
            import numpy as np
            labels = np.unique(y_true)
            try:
                # Ajustar columnas si faltan clases
                if y_score.shape[1] != len(labels):
                    import pandas as pd
                    y_score_df = pd.DataFrame(y_score)
                    y_score_df = y_score_df.reindex(columns=range(int(max(labels))+1), fill_value=0)
                    y_score = y_score_df.values
                auc = roc_auc_score(y_true, y_score, multi_class='ovr', labels=labels)
                self.log("test_auc", auc, prog_bar=True)
            except Exception:
                pass
        else:
            # AUC para binario
            try:
                if y_score.ndim > 1 and y_score.shape[1] > 1:
                    y_score_bin = y_score[:, 1]
                else:
                    y_score_bin = y_score
                auc = roc_auc_score(y_true, y_score_bin)
                self.log("test_auc", auc, prog_bar=True)
            except Exception:
                pass
        return loss


def model_build_fn(
        *,
        feature_keys: Sequence[str],
        label_key: str,
        label_names: Optional[Sequence[str]] = None,
        hidden_sizes: Optional[Sequence[int]] = None,
        activation: str = "relu",
        lr_rate: float = 1e-3,
        compression: Optional[str] = None,
) -> LightningModel:
    """
    Build an MLP LightningModel for tabular data.

    Args:
        feature_keys: list of feature column names.
        label_key: name of label column.
        label_names: if provided, len() determines num classes.
        hidden_sizes: sizes of hidden layers.
        activation: activation function for hidden layers.
        lr_rate: learning rate for optimizer.
        compression: optional compression method for model weights.
    """
    if label_names is not None:
        out_channels = len(label_names)
    else:
        out_channels = 1  # binary

    mlp = MLP(
        feature_keys=feature_keys,
        out_channels=out_channels,
        hidden_sizes=hidden_sizes,
        activation=activation,
        lr_rate=lr_rate,
        label_key=label_key,
    )
    return LightningModel(mlp, compression=compression)
