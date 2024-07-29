#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
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
"""Learning tests."""

from collections import OrderedDict

import torch

from p2pfl.learning.aggregators.fedavg import FedAvg
from p2pfl.learning.pytorch.lightning_learner import LightningLearner
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.node import Node
from p2pfl.utils import set_test_settings

set_test_settings()


###############################################################################
#    Test things related to the learning process (not the learning process)   #
###############################################################################


def test_encoding():
    """Test encoding and decoding of parameters."""
    nl1 = LightningLearner(MLP(), None, "", 1)
    encoded_params = nl1.encode_parameters()

    nl2 = LightningLearner(MLP(), None, "", 1)
    decoded_params = nl2.decode_parameters(encoded_params)
    nl2.set_parameters(decoded_params)

    assert encoded_params == nl2.encode_parameters()


def test_avg_simple():
    """Test simple aggregation (simple arrays)."""
    n = Node(None, None)
    n.start()
    aggregator = FedAvg()
    a = OrderedDict([("a", torch.tensor(-1)), ("b", torch.tensor(-1))])
    b = OrderedDict([("a", torch.tensor(0)), ("b", torch.tensor(0))])
    c = OrderedDict([("a", torch.tensor(1)), ("b", torch.tensor(1))])

    result = aggregator.aggregate({"a": (a, 1), "b": (b, 1), "c": (c, 1)})
    for layer in b:
        assert result[layer] == b[layer]

    result = aggregator.aggregate({"a": (a, 1), "b": (b, 7), "c": (c, 1)})
    for layer in b:
        assert result[layer] == b[layer]

    result = aggregator.aggregate({"a": (a, 800), "b": (b, 0), "c": (c, 0)})
    for layer in b:
        assert result[layer] == a[layer]

    n.stop()


def test_avg_complex():
    """Test complex aggregation (models)."""
    aggregator = FedAvg()
    nl1 = LightningLearner(MLP(), None, "", 1)
    params = nl1.get_parameters()
    params1 = nl1.get_parameters()
    params2 = nl1.get_parameters()

    result = aggregator.aggregate({"a": (params, 1)})
    # Check Results
    for layer in params:
        assert torch.eq(params[layer], result[layer]).all()

    for layer in params2:
        params1[layer] = params1[layer] + 1
        params2[layer] = params2[layer] - 1

    result = aggregator.aggregate({"a": (params1, 1), "b": (params2, 1)})

    # Check Results -> Careful with rounding errors
    for layer in params:
        a = torch.trunc(params[layer] * 10)
        b = torch.trunc(result[layer] * 10)
        assert torch.eq(a, b).all()
