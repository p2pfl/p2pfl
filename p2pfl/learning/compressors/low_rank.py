import numpy as np

from p2pfl.learning.compressors.compression_interface import CompressionStrategy


class LowRankApproximation(CompressionStrategy):
    """Low Rank compression strategy."""

    def apply_strategy(self, payload: dict, target_rank: int):
        """Compress the parameters."""
        params_to_share = []
        compressed_states = {}

        for pos, layer in enumerate(payload["params"]):
            if layer.ndim != 2 or target_rank < np.min(layer.shape):
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

        payload["params"] = params_to_share
        # si añades más cosas (e.g: PTQ), no se usarían en los compressed params, pero está bien así
        payload["additional_info"]["compressed_state"] = compressed_states
        return payload


    def reverse_strategy(self, payload: dict):
        """Decompress the parameters."""
        decompressed_params = []
        total_length = payload["params"].__len__() + payload["additional_info"]["compressed_state"].__len__()
        for pos in range(total_length):
            if pos in payload["additional_info"]["compressed_state"]:
                u, s, vt = payload["additional_info"]["compressed_state"][pos]
                decompressed_params.append(u @ np.diag(s) @ vt) # params approximation (np.ndarray)
            else:
                decompressed_params.append(payload["params"].pop(0))

        return decompressed_params


