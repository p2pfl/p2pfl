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
import json
import os
from typing import Optional, Tuple

import numpy as np

from p2pfl.learning.aggregators.aggregator import Aggregator, NoModelsToAggregateError
from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel
from p2pfl.learning.frameworks.xgboost.xgboost_model import XGBoostModel


# TODO: añadir mención a flower


class FedXgbBagging(Aggregator):
    """

    Paper: https://arxiv.org/abs/1602.05629.
    """

    SUPPORTS_PARTIAL_AGGREGATION: bool = True

    def __init__(self, disable_partial_aggregation: bool = False) -> None:
        """Initialize the aggregator."""
        super().__init__(disable_partial_aggregation=disable_partial_aggregation)

    def aggregate(self, models: list[XGBoostModel]) -> P2PFLModel:
        """
        Aggregate the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).

        Returns:
            A P2PFLModel with the aggregated.

        Raises:
            NoModelsToAggregateError: If there are no models to aggregate.

        """
        # Check if there are models to aggregate
        if len(models) == 0:
            raise NoModelsToAggregateError(f"({self.addr}) Trying to aggregate models when there is no models")

        # Total Samples
        total_samples = sum([m.get_num_samples() for m in models])

        # Add weighted models
        global_model = models[0].get_file_name()
        if global_model == "":
            return models[0]
        # Siempre cargar el JSON del primer modelo
        with open(global_model, "r") as f:
            global_model_json = json.load(f)
        os.remove(global_model)  # Remove the file to avoid conflicts
        if len(models) > 1:
            for m in models[1:]:
                model_file = m.get_file_name()
                with open(model_file, "r") as f:
                    current_model_json = json.load(f)
                os.remove(model_file)
                global_model_json = aggregate_boosters(global_model_json, current_model_json)

        # Save aggregated model to a temporary file
        global_model = global_model
        with open(global_model, "w") as f:
            json.dump(global_model_json, f)
        # Get contributors
        contributors: list[str] = []
        for m in models:
            contributors = contributors + m.get_contributors()

        # Return an aggregated p2pfl model
        returned_model = models[0].build_copy(params=[np.array(global_model)], num_samples=total_samples, contributors=contributors)
        # os.remove(global_model)  # Clean up the temporary file
        return returned_model


def _get_tree_nums(xgb_model: dict) -> Tuple[int, int]:
    # Get the number of trees
    tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_trees"
        ]
    )
    # Get the number of parallel trees
    paral_tree_num = int(
        xgb_model["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ]
    )
    return tree_num, paral_tree_num


def aggregate_boosters(
        bst_prev: Optional[dict],
        bst_curr: dict,
) -> dict:
    """Conduct bagging aggregation for given trees."""
    if not bst_prev:
        return bst_curr

    tree_num_prev, _ = _get_tree_nums(bst_prev)
    _, paral_tree_num_curr = _get_tree_nums(bst_curr)

    bst_prev["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
        "num_trees"
    ] = str(tree_num_prev + paral_tree_num_curr)
    iteration_indptr = bst_prev["learner"]["gradient_booster"]["model"][
        "iteration_indptr"
    ]
    bst_prev["learner"]["gradient_booster"]["model"]["iteration_indptr"].append(
        iteration_indptr[-1] + paral_tree_num_curr
    )

    # Aggregate new trees
    trees_curr = bst_curr["learner"]["gradient_booster"]["model"]["trees"]
    for tree_count in range(paral_tree_num_curr):
        trees_curr[tree_count]["id"] = tree_num_prev + tree_count
        bst_prev["learner"]["gradient_booster"]["model"]["trees"].append(
            trees_curr[tree_count]
        )
        bst_prev["learner"]["gradient_booster"]["model"]["tree_info"].append(0)

    return bst_prev
