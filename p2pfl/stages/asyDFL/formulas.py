import numpy as np


def compute_augmented_mixing_weight(out_neighbors, updating_nodes_at_k, dij, k, q, i):
    """
    Compute the [i, j] element of the augmented mixing weight matrix P̃k_q.

    Args:
        out_neighbors (dict): A dictionary where keys are node indices
                              and values are sets of their outgoing neighbors.
        updating_nodes_at_k (set): A set of indices of nodes that perform
                                   local updates in global iteration k (Ak).
        dij (dict): A dictionary of dictionaries where dij[i][j] represents the
                    global iteration interval between two successive P2P updates from j to i.
        k (int): The index of the current global iteration.
        q (int): The index of the augmented node (from 0 to dmax).
        i (int): The index of the edge node (node i).

    Returns:
        np.ndarray: A matrix of dimension (N, N) representing P̃k_q, where N is inferred from out_neighbors.

    """
    num_nodes = max(out_neighbors.keys()) + 1 if out_neighbors else 0
    P_kq = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    if k in updating_nodes_at_k and q == dij.get(i, {}).get(i, 0):
        num_out_neighbors_i = len(out_neighbors.get(i, set()))
        if num_out_neighbors_i > 0:
            for j in out_neighbors.get(i, set()):
                P_kq[i, j] = 1 / num_out_neighbors_i
    elif k not in updating_nodes_at_k and q == 0:
        P_kq[i, i] = 1.0

    return P_kq
