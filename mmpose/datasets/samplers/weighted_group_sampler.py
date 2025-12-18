# Copyright (c) OpenMMLab. All rights reserved.
"""Weighted sampler based on group_id for pose estimation datasets."""

import math
from typing import Iterator, Sized, Optional

import torch
from mmengine.dist import get_dist_info, sync_random_seed
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
        seed (int, optional): Random seed. If None, set a synchronized random seed.
            Default: None.
    """

    def __init__(self,
                 dataset: Sized,
                 group_id_weights: dict,
                 num_samples: int = None,
                 replacement: bool = True,
                 generator=None,
                 seed: Optional[int] = None,
                 **kwargs):
        if not hasattr(dataset, 'data_list'):
            raise ValueError(
                'Dataset must have data_list attribute to use WeightedGroupSampler'
            )

        self.dataset = dataset
        self.group_id_weights = group_id_weights
        self.replacement = replacement
        self.generator = generator

        # Get distributed info for multi-GPU training
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        # Initialize logger for debugging
        import logging
        logger = logging.getLogger('mmpose')

        # Ensure dataset is fully initialized before accessing data_list
        # This is important for lazy_init datasets
        if not dataset._fully_initialized:
            dataset.full_init()
        
        # Force reload data_list if it's empty (may happen in distributed training)
        # Check if data_list exists but is empty, which might indicate a loading issue
        if hasattr(dataset, 'data_list') and len(dataset.data_list) == 0:
            # Try to reload data_list by calling load_data_list again
            logger.warning(
                f'Dataset data_list is empty, attempting to reload. '
                f'Rank: {self.rank}, Dataset type: {type(dataset).__name__}'
            )
            
            try:
                # Reset initialization flag to force reload
                dataset._fully_initialized = False
                # Force reload by clearing cached data_list
                if hasattr(dataset, '_data_list'):
                    delattr(dataset, '_data_list')
                
                # Force full initialization again
                dataset.full_init()
                
                # CRITICAL FIX: If data_list is still empty after full_init(),
                # manually call load_data_list() and set _data_list directly
                if hasattr(dataset, 'data_list') and len(dataset.data_list) == 0:
                    logger.warning(
                        f'[Rank {self.rank}] data_list is still empty after full_init(), '
                        f'attempting manual reload...'
                    )
                    # Debug: Check _data_list before manual reload
                    has_data_list_attr = hasattr(dataset, '_data_list')
                    data_list_len_before = len(dataset._data_list) if has_data_list_attr else 0
                    logger.warning(
                        f'[Rank {self.rank}] Before manual reload: '
                        f'has _data_list: {has_data_list_attr}, '
                        f'_data_list length: {data_list_len_before}, '
                        f'data_list length: {len(dataset.data_list) if hasattr(dataset, "data_list") else 0}'
                    )
                    
                    # Manually load and set _data_list
                    data_list = dataset.load_data_list()
                    logger.warning(
                        f'[Rank {self.rank}] load_data_list() returned {len(data_list)} samples'
                    )
                    
                    if len(data_list) > 0:
                        dataset._data_list = data_list
                        dataset._fully_initialized = True
                        
                        # Debug: Verify _data_list was set correctly
                        has_data_list_attr_after = hasattr(dataset, '_data_list')
                        data_list_len_after = len(dataset._data_list) if has_data_list_attr_after else 0
                        data_list_prop_len = len(dataset.data_list) if hasattr(dataset, 'data_list') else 0
                        
                        logger.warning(
                            f'[Rank {self.rank}] After manual reload: '
                            f'has _data_list: {has_data_list_attr_after}, '
                            f'_data_list length: {data_list_len_after}, '
                            f'data_list property length: {data_list_prop_len}'
                        )
                        
                        if data_list_len_after > 0 and data_list_prop_len == 0:
                            logger.error(
                                f'[Rank {self.rank}] CRITICAL: _data_list was set ({data_list_len_after} items), '
                                f'but data_list property is still empty! This suggests a property caching issue.'
                            )
                        
                        logger.info(
                            f'[Rank {self.rank}] Successfully manually reloaded data_list. '
                            f'Data list length: {len(data_list)}'
                        )
                    else:
                        logger.error(
                            f'[Rank {self.rank}] load_data_list() returned empty list!'
                        )
                
                # Check if reload was successful
                if hasattr(dataset, 'data_list') and len(dataset.data_list) > 0:
                    logger.info(
                        f'Successfully reloaded data_list. '
                        f'Rank: {self.rank}, Data list length: {len(dataset.data_list)}'
                    )
                else:
                    logger.error(
                        f'Failed to reload data_list: still empty after reload. '
                        f'Rank: {self.rank}'
                    )
            except Exception as e:
                import traceback
                logger.error(
                    f'Failed to reload data_list: {e}\n'
                    f'Traceback: {traceback.format_exc()}'
                )
        
        # Calculate weights for each sample
        weights = []
        
        # CRITICAL: Try to access data_list directly, fallback to _data_list if property fails
        # This handles cases where data_list property might be cached or not working correctly
        data_list_to_use = None
        if hasattr(dataset, 'data_list'):
            try:
                data_list_to_use = dataset.data_list
                if len(data_list_to_use) > 0:
                    logger.info(
                        f'[Rank {self.rank}] Using data_list property: {len(data_list_to_use)} samples'
                    )
            except Exception as e:
                logger.warning(
                    f'[Rank {self.rank}] Failed to access data_list property: {e}, '
                    f'trying _data_list directly'
                )
        
        # Fallback to _data_list if data_list property is empty or fails
        if data_list_to_use is None or len(data_list_to_use) == 0:
            if hasattr(dataset, '_data_list'):
                data_list_to_use = dataset._data_list
                logger.warning(
                    f'[Rank {self.rank}] Using _data_list directly: {len(data_list_to_use) if data_list_to_use else 0} samples'
                )
        
        if data_list_to_use is None or len(data_list_to_use) == 0:
            # Provide more detailed error information
            import os
            dataset_info = {
                'type': type(dataset).__name__,
                'has_data_list': hasattr(dataset, 'data_list'),
                'fully_initialized': getattr(dataset, '_fully_initialized', False),
                'data_list_length': len(dataset.data_list) if hasattr(dataset, 'data_list') else 0,
            }
            # Try to get more info about the dataset
            if hasattr(dataset, 'ann_file'):
                ann_file = dataset.ann_file
                dataset_info['ann_file'] = ann_file
                # Check if annotation file exists
                if hasattr(dataset, 'data_root') and dataset.data_root:
                    full_ann_path = os.path.join(dataset.data_root, ann_file) if not os.path.isabs(ann_file) else ann_file
                    dataset_info['ann_file_exists'] = os.path.exists(full_ann_path)
                    dataset_info['ann_file_path'] = full_ann_path
                else:
                    dataset_info['ann_file_exists'] = os.path.exists(ann_file) if os.path.isabs(ann_file) else None
            if hasattr(dataset, 'data_root'):
                dataset_info['data_root'] = dataset.data_root
                dataset_info['data_root_exists'] = os.path.exists(dataset.data_root) if dataset.data_root else None
            
            # Add rank info for distributed training debugging
            dataset_info['rank'] = self.rank
            dataset_info['world_size'] = self.world_size
            # Add _data_list info
            dataset_info['has__data_list'] = hasattr(dataset, '_data_list')
            dataset_info['_data_list_length'] = len(dataset._data_list) if hasattr(dataset, '_data_list') else 0
            
            raise ValueError(
                f'Dataset data_list is empty! Cannot create WeightedGroupSampler. '
                f'Dataset info: {dataset_info}'
            )
        
        # Use the data_list we determined above
        for data_info in data_list_to_use:
            group_id = data_info.get('group_id')
            
            # Handle bottomup mode where group_id might be a list
            if isinstance(group_id, list):
                # For bottomup mode: use the first group_id that has a weight
                # or use the first group_id if none match
                weight = 1.0
                for gid in group_id:
                    if gid is not None and gid in group_id_weights:
                        weight = group_id_weights[gid]
                        break  # Use the first matching group_id
            elif group_id is not None and group_id in group_id_weights:
                # Topdown mode: single group_id
                weight = group_id_weights[group_id]
            else:
                weight = 1.0
            
            weights.append(weight)

        self.weights = torch.tensor(weights, dtype=torch.double)
        
        # Log dataset size for debugging
        if self.rank == 0:  # Only log from rank 0 to avoid duplicate logs
            import logging
            logger = logging.getLogger('mmpose')
            logger.info(
                f'WeightedGroupSampler initialized: dataset_size={len(weights)}, '
                f'num_samples={int(math.ceil(len(weights) * 1.0 / world_size))}, '
                f'world_size={world_size}, group_id_weights={group_id_weights}'
            )
        
        # Calculate num_samples for distributed training
        # In distributed training, each rank should sample approximately dataset_size / world_size samples
        # This is consistent with how DefaultSampler works in mmengine
        dataset_size = len(self.weights)
        if num_samples is not None:
            self.num_samples = num_samples
        else:
            # Each rank samples approximately dataset_size / world_size samples
            # Use ceil to ensure all ranks have enough samples
            self.num_samples = int(math.ceil(dataset_size * 1.0 / world_size))
        
        # Ensure num_samples > 0 to avoid RuntimeError
        if self.num_samples <= 0:
            self.num_samples = max(1, dataset_size)
        
        # Set up random seed for distributed training
        # Use synchronized seed across all ranks for consistent sampling
        self._base_seed = sync_random_seed() if seed is None else seed
        self.seed = self._base_seed

    def __iter__(self) -> Iterator[int]:
        """Generate an iterator of sample indices."""
        # Ensure num_samples > 0
        if self.num_samples <= 0:
            self.num_samples = max(1, len(self.weights))
        
        # Ensure weights is not empty
        if len(self.weights) == 0:
            return iter([])
        
        # For distributed training, we need to ensure each rank samples different data
        # Strategy: Each rank samples num_samples (dataset_size / world_size) samples independently
        # using rank-specific seed. This ensures each rank sees different samples while
        # maintaining weighted sampling distribution.
        
        # Create a rank-specific generator for independent sampling per rank
        # Each rank uses a different seed offset to ensure different samples
        rank_generator = torch.Generator()
        rank_generator.manual_seed(self.seed + self.rank)
        
        # Sample indices based on weights for this rank
        # Each rank samples num_samples (approximately dataset_size / world_size) samples
        sampled_indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=rank_generator).tolist()
        
        return iter(sampled_indices)

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible sampling across epochs."""
        # Update seed with epoch to ensure different samples each epoch
        # All ranks use the same epoch offset to maintain synchronization
        self.seed = self._base_seed + epoch * 1000

