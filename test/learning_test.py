from p2pfl.node import Node
from p2pfl.learning.agregators.fedavg import FedAvg
from p2pfl.learning.pytorch.mnist_examples.models.mlp import MLP
from p2pfl.learning.pytorch.lightninglearner import LightningLearner
from collections import OrderedDict
import torch

def test_encoding():
    nl1 = LightningLearner(MLP(), None)
    params = nl1.encode_parameters()

    nl2 = LightningLearner(MLP(), None)
    nl2.set_parameters(nl2.decode_parameters(params))

    params == nl2.encode_parameters()

def test_avg_simple():
    n = Node(None,None)
    n.start()
    agregator = FedAvg()
    a = OrderedDict([('a', torch.tensor(-1)), ('b', torch.tensor(-1))])
    b = OrderedDict([('a', torch.tensor(0)), ('b', torch.tensor(0))])
    c = OrderedDict([('a', torch.tensor(1)), ('b', torch.tensor(1))])

    result = agregator.agregate({"a":(a,1),"b":(b,1),"c":(c,1)})
    for layer in b:
        assert result[layer] == b[layer]

    result = agregator.agregate({"a":(a,1),"b":(b,7),"c":(c,1)}) 
    for layer in b:
        assert result[layer] == b[layer]

    result = agregator.agregate({"a":(a,800),"b":(b,0),"c":(c,0)})
    for layer in b:
        assert result[layer] == a[layer]

    n.stop()

def test_avg_complex():
    n = Node(None,None)
    n.start()

    agregator = FedAvg()
    nl1 = LightningLearner(MLP(), None)
    params = nl1.get_parameters()
    params1 = nl1.get_parameters()
    params2 = nl1.get_parameters()

    result = agregator.agregate({"a":(params,1)})
    # Check Results
    for layer in params:
        assert torch.eq(params[layer], result[layer]).all()

    for layer in params2:
        params1[layer] = params1[layer]+1
        params2[layer] = params2[layer]-1
    
    result = agregator.agregate({"a":(params1,1),  "b":(params2,1)}) 

    # Check Results -> Careful with rounding errors
    for layer in params:
        a = torch.trunc(params[layer]*10)
        b = torch.trunc(result[layer]*10)
        assert torch.eq(a, b).all()

    n.stop()