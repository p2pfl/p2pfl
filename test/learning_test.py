from p2pfl.learning.agregators.fedavg import FedAvg
from p2pfl.learning.pytorch.models.mlp import MLP
from p2pfl.learning.pytorch.learners.lightninglearner import LightningLearner
from collections import OrderedDict
import torch

def test_encoding():
    nl1 = LightningLearner(MLP(), None)
    params = nl1.encode_parameters()

    nl2 = LightningLearner(MLP(), None)
    nl2.set_parameters(nl2.decode_parameters(params))

    params == nl2.encode_parameters()

def test_avg_simple():
    a = OrderedDict([('a', torch.tensor(-1)), ('b', torch.tensor(-1))])
    b = OrderedDict([('a', torch.tensor(0)), ('b', torch.tensor(0))])
    c = OrderedDict([('a', torch.tensor(1)), ('b', torch.tensor(1))])

    result = FedAvg.agregate([(a,1),(b,1),(c,1)])
    for layer in b:
        assert result[layer] == b[layer]

    result = FedAvg.agregate([(a,1),(b,7),(c,1)])
    for layer in b:
        assert result[layer] == b[layer]

    result = FedAvg.agregate([(a,800),(b,0),(c,0)])
    for layer in b:
        assert result[layer] == a[layer]

def test_avg_complex():
    nl1 = LightningLearner(MLP(), None)
    params = nl1.get_parameters()
    params1 = nl1.get_parameters()
    params2 = nl1.get_parameters()

    result = FedAvg.agregate([(params,1)])

    # Check Results
    for layer in params:
        assert torch.eq(params[layer], result[layer]).all()

    for layer in params2:
        params1[layer] = params1[layer]+1
        params2[layer] = params2[layer]-1
    
    result = FedAvg.agregate([(params1,1), (params2,1)])

    # Check Results -> Careful with rounding errors
    for layer in params:
        a = torch.trunc(params[layer]*10)
        b = torch.trunc(result[layer]*10)
        assert torch.eq(a, b).all()
