# Copyright (c) OpenMMLab. All rights reserved.
"""Transform to apply group_id-based sample weights to heatmap weights."""

import numpy as np
from mmengine.registry import TRANSFORMS

from .formatting import BaseTransform


@TRANSFORMS.register_module()
class ApplyGroupWeight(BaseTransform):
    """Apply sample weights based on group_id to heatmap weights.

    This transform multiplies heatmap_weights and displacement_weights
    by the sample_weight for each instance, allowing different group_ids
    to have different loss weights during training.

    Required Keys:
        - heatmap_weights (np.ndarray): Heatmap weights in shape (N, K, H, W)
        - displacement_weights (np.ndarray, optional): Displacement weights
        - instance_weights (list, optional): Per-instance weights from dataset

    Modified Keys:
        - heatmap_weights: Multiplied by instance_weights
        - displacement_weights: Multiplied by instance_weights (if exists)
    """

    def transform(self, results: dict) -> dict:
        """Apply instance weights to heatmap and displacement weights.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: Updated result dict with weighted heatmap/displacement weights.
        """
        if 'instance_weights' not in results:
            # No instance weights, return as is
            return results

        instance_weights = results['instance_weights']
        if not isinstance(instance_weights, (list, np.ndarray)):
            return results

        instance_weights = np.array(instance_weights, dtype=np.float32)

        # Apply weights to heatmap_weights
        if 'heatmap_weights' in results:
            heatmap_weights = results['heatmap_weights']
            if isinstance(heatmap_weights, np.ndarray):
                # heatmap_weights shape: (N, K, H, W) or (K, H, W)
                if heatmap_weights.ndim == 4:
                    # (N, K, H, W) - multiply each instance
                    for i, weight in enumerate(instance_weights):
                        if i < heatmap_weights.shape[0]:
                            heatmap_weights[i] = heatmap_weights[i] * weight
                elif heatmap_weights.ndim == 3:
                    # (K, H, W) - single instance, multiply by first weight
                    if len(instance_weights) > 0:
                        heatmap_weights = heatmap_weights * instance_weights[0]
                results['heatmap_weights'] = heatmap_weights

        # Apply weights to displacement_weights
        if 'displacement_weights' in results:
            displacement_weights = results['displacement_weights']
            if isinstance(displacement_weights, np.ndarray):
                # displacement_weights shape: (N, K*2, H, W) or (K*2, H, W)
                if displacement_weights.ndim == 4:
                    # (N, K*2, H, W) - multiply each instance
                    for i, weight in enumerate(instance_weights):
                        if i < displacement_weights.shape[0]:
                            displacement_weights[i] = displacement_weights[i] * weight
                elif displacement_weights.ndim == 3:
                    # (K*2, H, W) - single instance, multiply by first weight
                    if len(instance_weights) > 0:
                        displacement_weights = displacement_weights * instance_weights[0]
                results['displacement_weights'] = displacement_weights

        return results

    def __repr__(self) -> str:
        """Print the basic information of the transform."""
        return self.__class__.__name__ + '()'

