# Copyright (c) OpenMMLab. All rights reserved.
"""Weighted sampler based on group_id for pose estimation datasets."""

from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler

from mmpose.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class WeightedGroupSampler(Sampler):
    """Weighted sampler that samples based on group_id.

    This sampler increases the sampling probability of samples with specific
    group_ids, allowing more frequent training on these samples.

    Args:
        dataset (Sized): The dataset to sample from. Must have 'group_id'
            attribute in each data sample.
        group_id_weights (dict): Dictionary mapping group_id to sampling weight.
            For example, {1: 2.0} means group_id=1 samples will be sampled
            2 times more frequently.
        num_samples (int, optional): Number of samples to draw. If None,
            defaults to len(dataset).
        replacement (bool): Whether to sample with replacement. Default: True.
        generator: Generator used for random sampling. Default: None.
    """

    def __init__(self,
                 dataset: Sized,
                 group_id_weights: dict,
                 num_samples: int = None,
                 replacement: bool = True,
                 generator=None):
        if not hasattr(dataset, 'data_list'):
            raise ValueError(
                'Dataset must have data_list attribute to use WeightedGroupSampler'
            )

        self.dataset = dataset
        self.group_id_weights = group_id_weights
        self.replacement = replacement
        self.generator = generator

        # Calculate weights for each sample
        weights = []
        for data_info in dataset.data_list:
            group_id = data_info.get('group_id')
            if group_id is not None and group_id in group_id_weights:
                weight = group_id_weights[group_id]
            else:
                weight = 1.0
            weights.append(weight)

        self.weights = torch.tensor(weights, dtype=torch.double)
        self.num_samples = num_samples if num_samples is not None else len(
            self.weights)

    def __iter__(self) -> Iterator[int]:
        """Generate an iterator of sample indices."""
        return iter(
            torch.multinomial(
                self.weights,
                self.num_samples,
                self.replacement,
                generator=self.generator).tolist())

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples

