#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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

"""Federated Averaging (FedAvg) Aggregator."""



from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.xgboost.xgboost_model import XGBoostModel

# TODO: añadir mención a flower


class FedXgbCyclic(Aggregator):
    """Paper: https://arxiv.org/abs/1602.05629."""

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

    def aggregate(self, models: list[XGBoostModel]) -> P2PFLModel:
        """Cyclic aggregation: solo un cliente participa por ronda, el modelo se pasa secuencialmente."""
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")

        # En entrenamiento cíclico, solo se toma el modelo del cliente actual (el último de la lista)
        selected_model = models[-1]
        total_samples = selected_model.get_num_samples()
        contributors = selected_model.get_contributors()
        return selected_model.build_copy(params=selected_model.get_parameters(), num_samples=total_samples, contributors=contributors)
