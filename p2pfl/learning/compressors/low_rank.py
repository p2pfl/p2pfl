import numpy as np

from p2pfl.learning.compressors.compression_interface import CompressionStrategy


class LowRank(CompressionStrategy):
    """Low Rank compression strategy."""

    def apply_strategy(self, data: dict, target_rank: int):
        """Compress the parameters."""
        params_to_share = []
        compressed_states = {}

        for pos, layer in enumerate(data["params"]):
            if  layer.ndim != 2 or target_rank < np.min(layer.shape):
                # TODO: Logger? no puedo acceder a node name desde P2PFLModel
                # TODO: svd solo se puede aplicar sobre matrices 2D
                # puedo redimensionar la matriz para que sea 2D,
                # pero depende de la implementación así que mejor no
                params_to_share.append(layer) # list(np.ndarray)
            else:
                u, s, vt = np.linalg.svd(layer, full_matrices=False)
                u, s, vt = u[:, :target_rank], s[0:target_rank, :], vt[:target_rank, :]
                # remove layer from data and store compressed layer
                compressed_states[pos] = u, s, vt

        data["params"] = params_to_share
        # si añades más cosas (e.g: PTQ), no se usarían en los compressed params, pero está bien así
        data["additional_info"]["compressed_state"] = compressed_states
        return data


    def reverse_strategy(self, data: dict):
        """Decompress the parameters."""
        decompressed_params = []
        total_length = data["params"].__len__() + data["additional_info"]["compressed_state"].__len__()
        for pos in range(total_length):
            if pos in data["additional_info"]["compressed_state"]:
                u, s, vt = data["additional_info"]["compressed_state"][pos]
                decompressed_params.append(u @ s @ vt) # params approximation (np.ndarray)
            else:
                decompressed_params.append(data["params"].pop(0))

        return decompressed_params


