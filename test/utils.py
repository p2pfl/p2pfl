import time

import torch

def wait_network_nodes(nodes):
    acum = 0
    while True:
        begin = time.time()
        if all([ len(n.get_network_nodes()) == len(nodes) for n in nodes]):
            break
        time.sleep(0.1)
        acum += time.time() - begin
        if acum > 6:
            assert False

def wait_4_results(nodes):
    while True:
        time.sleep(1)
        finish = True
        for f in [node.round is None for node in nodes]:
            finish = finish and f

        if finish:
            break

def check_equal_models(nodes):
    model = None
    first = True
    for node in nodes:
        if first:
            model = node.learner.get_parameters()
            first = False
        else:
            for layer in model:
                a = torch.round(model[layer], decimals=2)
                b = torch.round(node.learner.get_parameters()[layer], decimals=2)
                assert torch.eq(a, b).all()