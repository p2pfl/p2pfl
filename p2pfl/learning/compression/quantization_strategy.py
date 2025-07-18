#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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
# along with
# this program. If not, see <http://www.gnu.org/licenses/>.
#
#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2024 Pedro Guijas Bravo.
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
# along with
# this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Post-Training Quantization (PTQ) compression strategy."""

from typing import Literal

import numpy as np

from p2pfl.learning.compression.base_compression_strategy import TensorCompressor


class PTQuantization(TensorCompressor):
    """Post-Training Quantization (PTQ) with proper scaling."""

    #
    # DEVELOPERS! -> very careful with overflow
    #

    # Define supported data types
    _SUPPORTED_FLOAT_TYPES = {np.float16, np.float32, np.float64}
    _SUPPORTED_INT_TYPES = {np.int8, np.uint8, np.int16, np.uint16, np.int32}

    def apply_strategy(
        self,
        params: list[np.ndarray],
        dtype: str = "float16",
        scheme: Literal["symmetric", "asymmetric"] = "symmetric",
        granularity: Literal["per_tensor", "per_channel"] = "per_tensor",
        channel_axis: int = 0,
    ) -> tuple[list[np.ndarray], dict]:
        """
        Reduce the precision of model parameters with proper scaling.

        Args:
            params: The parameters to compress.
            dtype: The desired precision (e.g., "float16", "int8").
            scheme: Quantization scheme - "symmetric" (centered around 0) or "asymmetric" (uses full range).
            granularity: "per_tensor" uses one scale for the whole tensor, "per_channel" uses separate scales for each channel.
            channel_axis: Axis to use for per-channel quantization.

        Returns:
            Tuple of quantized parameters and additional info for dequantization.

        Raises:
            ValueError: If an unsupported data type is provided or if parameters are invalid.

        """
        if not params:
            raise ValueError("Empty parameter list provided for quantization")

        # Parse and validate the target dtype
        try:
            target_dtype = np.dtype(dtype)
        except TypeError as e:
            raise ValueError(f"Invalid dtype: {dtype}") from e

        # Check if the target dtype is supported
        if np.issubdtype(target_dtype, np.floating):
            if target_dtype.type not in self._SUPPORTED_FLOAT_TYPES:
                raise ValueError(f"Unsupported float dtype: {dtype}. Supported float types: {self._SUPPORTED_FLOAT_TYPES}")
        elif np.issubdtype(target_dtype, np.integer):
            if target_dtype.type not in self._SUPPORTED_INT_TYPES:
                raise ValueError(f"Unsupported integer dtype: {dtype}. Supported integer types: {self._SUPPORTED_INT_TYPES}")
        else:
            raise ValueError(f"Unsupported dtype for quantization: {dtype}. Only floating point and integer types are supported.")

        # Validate scheme
        if scheme not in ["symmetric", "asymmetric"]:
            raise ValueError(f"Unsupported quantization scheme: {scheme}. Supported schemes are 'symmetric' and 'asymmetric'.")

        # Validate granularity
        if granularity not in ["per_tensor", "per_channel"]:
            raise ValueError(
                f"Unsupported quantization granularity: {granularity}. Supported granularities are 'per_tensor' and 'per_channel'."
            )

        original_dtype = params[0].dtype

        # If target dtype is floating point, just do simple cast
        if np.issubdtype(target_dtype, np.floating):
            return [param.astype(target_dtype) for param in params], {
                "ptq_original_dtype": original_dtype,
                "ptq_type": "float",  # Add quantization type for clarity
            }

        # For integer quantization, we need to scale properly
        quantized_params = []
        scales: list[float | np.ndarray] = []  # Updated typing to allow both scalar and array
        zero_points: list[int | np.ndarray] = []  # Updated typing to allow both scalar and array

        for param in params:
            if granularity == "per_tensor":
                # Quantize the entire tensor with a single scale
                q_param, scale, zero_point = self._quantize_tensor(param, target_dtype, scheme)
                scales.append(scale)
                zero_points.append(zero_point)
            else:  # per_channel
                # Quantize each channel separately
                q_param, channel_scales, channel_zero_points = self._quantize_per_channel(param, target_dtype, scheme, channel_axis)
                scales.append(channel_scales)
                zero_points.append(channel_zero_points)

            quantized_params.append(q_param)

        return quantized_params, {
            "ptq_original_dtype": original_dtype,
            "ptq_type": "int",  # Add quantization type for clarity
            "ptq_scheme": scheme,
            "ptq_granularity": granularity,
            "ptq_scales": scales,
            "ptq_zero_points": zero_points,
            "ptq_channel_axis": channel_axis if granularity == "per_channel" else None,
        }

    def reverse_strategy(self, params: list[np.ndarray], additional_info: dict) -> list[np.ndarray]:
        """
        Return model parameters to saved original precision.

        Args:
            params: The parameters to decompress.
            additional_info: Additional information for decompression.

        Returns:
            Decompressed parameters.

        Raises:
            ValueError: If the parameters or additional info are invalid.

        """
        if not params:
            raise ValueError("Empty parameter list provided for dequantization")

        if "ptq_original_dtype" not in additional_info:
            raise ValueError("Missing 'ptq_original_dtype' in additional_info")
        original_dtype = additional_info["ptq_original_dtype"]

        # If floating point quantization was used (simple cast)
        if additional_info.get("ptq_type") == "float":
            return [param.astype(original_dtype) for param in params]

        # For integer quantization, we need proper dequantization
        required_keys = ["ptq_scheme", "ptq_granularity", "ptq_scales", "ptq_zero_points"]
        for key in required_keys:
            if key not in additional_info:
                raise ValueError(f"Missing required key '{key}' in additional_info")

        scheme = additional_info["ptq_scheme"]
        granularity = additional_info["ptq_granularity"]
        scales = additional_info["ptq_scales"]
        zero_points = additional_info["ptq_zero_points"]

        # Validate scheme and granularity
        if scheme not in ["symmetric", "asymmetric"]:
            raise ValueError(f"Unsupported quantization scheme: {scheme}")

        if granularity not in ["per_tensor", "per_channel"]:
            raise ValueError(f"Unsupported quantization granularity: {granularity}")

        # Get channel_axis for per_channel quantization
        channel_axis = additional_info["ptq_channel_axis"]
        if granularity == "per_channel" and channel_axis is None:
            raise ValueError("Missing 'ptq_channel_axis' for per_channel dequantization")

        dequantized_params = []

        for i, param in enumerate(params):
            if i >= len(scales) or i >= len(zero_points):
                raise ValueError(f"Not enough scale/zero_point values for parameter at index {i}")

            if granularity == "per_tensor":
                # Dequantize the entire tensor with a single scale
                dq_param = self._dequantize_tensor(param, scales[i], zero_points[i], original_dtype)
            else:  # per_channel
                # Dequantize each channel separately
                dq_param = self._dequantize_per_channel(param, scales[i], zero_points[i], channel_axis, original_dtype)

            dequantized_params.append(dq_param)

        return dequantized_params

    def _quantize_tensor(self, tensor: np.ndarray, target_dtype: np.dtype, scheme: str) -> tuple[np.ndarray, float, int]:
        """
        Quantize a tensor with a single scale factor.

        Args:
            tensor: The tensor to quantize.
            target_dtype: The target data type.
            scheme: Quantization scheme ("symmetric" or "asymmetric").

        Returns:
            Tuple of (quantized_tensor, scale, zero_point).

        Raises:
            ValueError: If the target dtype is not supported.

        """
        # Determine quantization range based on target dtype
        if target_dtype == np.int8:
            qmin, qmax = -128, 127
        elif target_dtype == np.uint8:
            qmin, qmax = 0, 255
        elif target_dtype == np.int16:
            qmin, qmax = -32768, 32767
        elif target_dtype == np.uint16:
            qmin, qmax = 0, 65535
        elif target_dtype == np.int32:
            qmin, qmax = -2147483648, 2147483647
        else:
            raise ValueError(f"Unsupported integer dtype for quantization: {target_dtype}")

        # Handle empty tensor
        if tensor.size == 0:
            return np.array([], dtype=target_dtype), 1.0, 0

        if scheme == "symmetric":
            # Symmetric quantization (centered around 0)
            abs_max = max(abs(tensor.min()), abs(tensor.max()))

            # Avoid division by zero
            scale = 1.0 if abs_max == 0 else abs_max / max(abs(qmin), abs(qmax))
            zero_point = 0

            # FIX: First clip the raw floating-point values before converting to integer
            # This prevents the overflow issue where 128 wraps around to -128
            raw_quantized = np.round(tensor / scale)

            # Important: Clip BEFORE converting to int type to prevent overflow
            raw_quantized_clipped = np.clip(raw_quantized, qmin, qmax)

            # Now safely convert to target dtype
            quantized = raw_quantized_clipped.astype(target_dtype)

        else:  # asymmetric
            # Asymmetric quantization (uses full range)
            tmin, tmax = tensor.min(), tensor.max()

            # Handle the case where min equals max (constant tensor)
            if tmin == tmax:
                if tmin == 0:
                    return np.zeros_like(tensor, dtype=target_dtype), 1.0, 0
                else:
                    # Map the constant value to the middle of the quantization range
                    mid_q = (qmin + qmax) // 2
                    return np.full_like(tensor, mid_q, dtype=target_dtype), tmin / mid_q, 0

            # Calculate scale and zero point
            scale = (tmax - tmin) / (qmax - qmin)
            zero_point = qmin - round(tmin / scale)
            zero_point = max(qmin, min(qmax, zero_point))  # Clamp zero_point

            # FIX: Similar fix for asymmetric quantization
            raw_quantized = np.round(tensor / scale + zero_point)
            raw_quantized_clipped = np.clip(raw_quantized, qmin, qmax)
            quantized = raw_quantized_clipped.astype(target_dtype)

        return quantized, float(scale), int(zero_point)

    def _quantize_per_channel(
        self, tensor: np.ndarray, target_dtype: np.dtype, scheme: str, channel_axis: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quantize a tensor with per-channel scale factors.

        Args:
            tensor: The tensor to quantize.
            target_dtype: The target data type.
            scheme: Quantization scheme ("symmetric" or "asymmetric").
            channel_axis: The axis along which to compute per-channel scales.

        Returns:
            Tuple of (quantized_tensor, scales, zero_points).

        Raises:
            ValueError: If the target dtype is not supported or if channel_axis is invalid.

        """
        # Validate channel_axis
        if tensor.ndim == 0:
            raise ValueError("Cannot perform per-channel quantization on a scalar tensor")

        if not 0 <= channel_axis < tensor.ndim:
            raise ValueError(f"Invalid channel_axis {channel_axis} for tensor with {tensor.ndim} dimensions")

        # Only apply per-channel quantization if tensor has enough dimensions
        if tensor.ndim <= 1:
            # Fall back to per-tensor quantization for 1D tensors
            q_tensor, scale, zero_point = self._quantize_tensor(tensor, target_dtype, scheme)
            return q_tensor, np.array([scale], dtype=np.float32), np.array([zero_point], dtype=np.int32)

        # Determine quantization range based on target dtype
        if target_dtype == np.int8:
            qmin, qmax = -128, 127
        elif target_dtype == np.uint8:
            qmin, qmax = 0, 255
        elif target_dtype == np.int16:
            qmin, qmax = -32768, 32767
        elif target_dtype == np.uint16:
            qmin, qmax = 0, 65535
        elif target_dtype == np.int32:
            qmin, qmax = -2147483648, 2147483647
        else:
            raise ValueError(f"Unsupported integer dtype for quantization: {target_dtype}")

        # Get number of channels
        num_channels = tensor.shape[channel_axis]

        # Handle empty tensor
        if num_channels == 0 or tensor.size == 0:
            return (np.array([], dtype=target_dtype), np.array([], dtype=np.float32), np.array([], dtype=np.int32))

        # Initialize arrays for scales and zero points
        scales = np.zeros(num_channels, dtype=np.float32)
        zero_points = np.zeros(num_channels, dtype=np.int32)

        # Create a view where the channel dimension is the first dimension
        transposed_tensor = np.moveaxis(tensor, channel_axis, 0)

        # Initialize quantized tensor with the same shape as the input tensor
        quantized = np.zeros_like(tensor, dtype=target_dtype)
        transposed_quantized = np.moveaxis(quantized, channel_axis, 0)

        # Quantize each channel separately
        for c in range(num_channels):
            channel_tensor = transposed_tensor[c]

            if scheme == "symmetric":
                # Handle empty or constant channel
                if channel_tensor.size == 0 or (channel_tensor.min() == channel_tensor.max() == 0):
                    scales[c] = 1.0
                    zero_points[c] = 0
                    continue

                # Symmetric quantization (centered around 0)
                abs_max = max(abs(channel_tensor.min()), abs(channel_tensor.max()))

                # Avoid division by zero
                scale = 1.0 if abs_max == 0 else abs_max / max(abs(qmin), abs(qmax))
                zero_point = 0

                # Quantize
                raw_quantized = np.round(channel_tensor / scale)
                channel_quantized = np.clip(raw_quantized, qmin, qmax).astype(target_dtype)

                # Clamp to ensure we're in the valid range
                channel_quantized = np.clip(channel_quantized, qmin, qmax)

            else:  # asymmetric
                # Asymmetric quantization (uses full range)
                tmin, tmax = channel_tensor.min(), channel_tensor.max()

                # Handle the case where min equals max (constant tensor)
                if tmin == tmax:
                    if tmin == 0:
                        transposed_quantized[c] = np.zeros_like(channel_tensor, dtype=target_dtype)
                        scales[c] = 1.0
                        zero_points[c] = 0
                        continue
                    else:
                        # Map the constant value to the middle of the quantization range
                        mid_q = (qmin + qmax) // 2
                        transposed_quantized[c] = np.full_like(channel_tensor, mid_q, dtype=target_dtype)
                        scales[c] = float(tmin / mid_q)
                        zero_points[c] = 0
                        continue

                # Calculate scale and zero point
                scale = (tmax - tmin) / (qmax - qmin)

                # Avoid division by zero
                if scale == 0:
                    scale = 1.0

                zero_point = qmin - round(tmin / scale)
                zero_point = max(qmin, min(qmax, zero_point))  # Clamp zero_point

                # Quantize
                channel_quantized = np.round(channel_tensor / scale + zero_point).astype(target_dtype)

                # Clamp to ensure we're in the valid range
                channel_quantized = np.clip(channel_quantized, qmin, qmax)

            # Store the quantized channel
            transposed_quantized[c] = channel_quantized

            # Store scale and zero point for this channel
            scales[c] = float(scale)
            zero_points[c] = int(zero_point)

        return quantized, scales, zero_points

    def _dequantize_tensor(self, tensor: np.ndarray, scale: float, zero_point: int, original_dtype: np.dtype) -> np.ndarray:
        """
        Dequantize a tensor with a single scale factor.

        Args:
            tensor: The quantized tensor.
            scale: The scale factor used for quantization.
            zero_point: The zero point used for quantization.
            original_dtype: The original data type to cast back to.

        Returns:
            Dequantized tensor.

        Raises:
            ValueError: If the scale or zero_point are invalid.

        """
        # Validate scale and zero_point
        if not isinstance(scale, int | float):
            raise ValueError(f"Invalid scale factor: {scale}. Scale must be a number.")

        if scale <= 0:
            raise ValueError(f"Invalid scale factor: {scale}. Scale must be positive.")

        if not isinstance(zero_point, int | np.integer):
            raise ValueError(f"Invalid zero point: {zero_point}. Zero point must be an integer.")

        # Handle empty tensor
        if tensor.size == 0:
            return np.array([], dtype=original_dtype)

        # Dequantize
        if zero_point == 0:  # noqa: SIM108
            # Symmetric dequantization
            dequantized = tensor.astype(np.float32) * scale
        else:
            # Asymmetric dequantization
            dequantized = (tensor.astype(np.float32) - zero_point) * scale

        # Cast back to original dtype
        return dequantized.astype(original_dtype)

    def _dequantize_per_channel(
        self, tensor: np.ndarray, scales: np.ndarray, zero_points: np.ndarray, channel_axis: int, original_dtype: np.dtype
    ) -> np.ndarray:
        """
        Dequantize a tensor with per-channel scale factors.

        Args:
            tensor: The quantized tensor.
            scales: The scale factors used for quantization.
            zero_points: The zero points used for quantization.
            channel_axis: The axis along which per-channel quantization was applied.
            original_dtype: The original data type to cast back to.

        Returns:
            Dequantized tensor.

        Raises:
            ValueError: If the scales, zero_points, or channel_axis are invalid.

        """
        # Validate inputs
        if not isinstance(scales, np.ndarray):
            raise ValueError(f"Scales must be a numpy array, got {type(scales)}")

        if not isinstance(zero_points, np.ndarray):
            raise ValueError(f"Zero points must be a numpy array, got {type(zero_points)}")

        if scales.ndim != 1:
            raise ValueError(f"Scales must be a 1D array, got {scales.ndim}D array")

        if zero_points.ndim != 1:
            raise ValueError(f"Zero points must be a 1D array, got {zero_points.ndim}D array")

        if len(scales) != len(zero_points):
            raise ValueError(f"Number of scales ({len(scales)}) must match number of zero points ({len(zero_points)})")

        # Handle empty tensor
        if tensor.size == 0 or scales.size == 0:
            return np.array([], dtype=original_dtype)

        # If scales is a single value, use per-tensor dequantization
        if scales.size == 1:
            return self._dequantize_tensor(tensor, scales[0], zero_points[0], original_dtype)

        # Validate channel_axis
        if tensor.ndim == 0:
            raise ValueError("Cannot perform per-channel dequantization on a scalar tensor")

        if not 0 <= channel_axis < tensor.ndim:
            raise ValueError(f"Invalid channel_axis {channel_axis} for tensor with {tensor.ndim} dimensions")

        # Check if the number of channels matches the number of scales
        num_channels = tensor.shape[channel_axis]
        if num_channels != len(scales):
            raise ValueError(f"Number of channels ({num_channels}) must match number of scales ({len(scales)})")

        # Create a view where the channel dimension is the first dimension
        transposed_tensor = np.moveaxis(tensor, channel_axis, 0)

        # Initialize dequantized tensor with the same shape as the input tensor but float32 dtype
        dequantized = np.zeros_like(tensor, dtype=np.float32)
        transposed_dequantized = np.moveaxis(dequantized, channel_axis, 0)

        # Dequantize each channel separately
        for c in range(len(scales)):
            channel_tensor = transposed_tensor[c]
            scale = scales[c]
            scale_value = float(scale)
            zero_point = zero_points[c]

            # Validate scale and zero_point
            if not np.isscalar(scale) or not np.isfinite(scale):
                raise ValueError(f"Invalid scale factor at index {c}: {scale_value}. Scale must be a finite number.")

            if scale_value <= 0:
                raise ValueError(f"Invalid scale factor at index {c}: {scale_value}. Scale must be positive.")

            if not isinstance(zero_point, int | np.integer):
                raise ValueError(f"Invalid zero point at index {c}: {zero_point}. Zero point must be an integer.")

            # Dequantize
            if zero_point == 0:
                # Symmetric dequantization
                channel_dequantized = channel_tensor.astype(np.float32) * scale
            else:
                # Asymmetric dequantization
                channel_dequantized = (channel_tensor.astype(np.float32) - zero_point) * scale

            # Store the dequantized channel
            transposed_dequantized[c] = channel_dequantized

        # Cast back to original dtype
        return dequantized.astype(original_dtype)
