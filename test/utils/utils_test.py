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

"""Utility functions testing."""

import pytest

from p2pfl.learning.dataset.p2pfl_dataset import DatasetEmptyException, P2PFLEmptyDataset
from p2pfl.learning.frameworks.p2pfl_model import ModelEmptyException, P2PFLEmptyModel


def test_raise_on_model_access():
    """Test that accessing an attribute raises ModelEmptyException."""
    empty_model = P2PFLEmptyModel()
    with pytest.raises(ModelEmptyException):
        print(empty_model.model)


def test_method_call_raises_modelemptyexception():
    """Test that calling a method raises ModelEmptyException."""
    empty_dataset = P2PFLEmptyDataset()
    with pytest.raises(DatasetEmptyException):
        print(empty_dataset._data)
