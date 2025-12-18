# Copyright (c) OpenMMLab. All rights reserved.
import json
from pathlib import Path
from typing import Dict, List, Optional

from mmpose.registry import DATASETS
from ..base import BaseCocoStyleDataset


@DATASETS.register_module()
class CocoParallelDataset(BaseCocoStyleDataset):
    """COCO Parallel dataset for pose estimation.

    Custom dataset for parallel bar pose estimation with 21 keypoints.
    Based on COCO format with 4 additional keypoints for feet.

    COCO Parallel keypoints::

        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle',
        17: 'left_heel',      # 新增
        18: 'right_heel',     # 新增
        19: 'left_foot',      # 新增
        20: 'right_foot'      # 新增

    Args:
        ann_file (str): Annotation file path. Default: ''.
        bbox_file (str, optional): Detection result file path. If
            ``bbox_file`` is set, detected bboxes loaded from this file will
            be used instead of ground-truth bboxes. This setting is only for
            evaluation, i.e., ignored when ``test_mode`` is ``False``.
            Default: ``None``.
        data_mode (str): Specifies the mode of data samples: ``'topdown'`` or
            ``'bottomup'``. In ``'topdown'`` mode, each data sample contains
            one instance; while in ``'bottomup'`` mode, each data sample
            contains all instances in a image. Default: ``'topdown'``
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Default: ``None``.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Default: ``None``.
        data_prefix (dict, optional): Prefix for training data. Default:
            ``dict(img=None, ann=None)``.
        filter_cfg (dict, optional): Config for filter data. Default: `None`.
        indices (int or Sequence[int], optional): Support using first few
            data in annotation file to facilitate training/testing on a smaller
            dataset. Default: ``None`` which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
            Default: ``True``.
        pipeline (list, optional): Processing pipeline. Default: [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Default: ``False``.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Default: ``False``.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Default: 1000.
    """

    METAINFO: dict = dict(from_file='configs/_base_/datasets/coco_parallel.py')

    def __init__(self,
                 ann_file: str = '',
                 group_id_mapping_file: Optional[str] = None,
                 group_id_weight: Optional[dict] = None,
                 **kwargs):
        """Initialize CocoParallelDataset.

        Args:
            ann_file (str): Annotation file path.
            group_id_mapping_file (str, optional): Path to JSON file mapping
                annotation IDs to group_ids. Format: {ann_id: group_id}.
                If None, group_id will not be used. Default: None.
            group_id_weight (dict, optional): Weight multipliers for different
                group_ids. Format: {group_id: weight}. For example,
                {1: 2.0} means group_id=1 samples will have 2x weight.
                Default: None.
            **kwargs: Other arguments passed to BaseCocoStyleDataset.
        """
        self.group_id_mapping_file = group_id_mapping_file
        self.group_id_weight = group_id_weight or {}
        self.group_id_mapping = {}
        
        # Load group_id mapping if provided
        if group_id_mapping_file:
            group_id_mapping_path = Path(group_id_mapping_file)
            if not group_id_mapping_path.is_absolute():
                # If relative path, try to resolve relative to ann_file
                ann_file_path = Path(ann_file)
                if ann_file_path.is_absolute():
                    group_id_mapping_path = ann_file_path.parent / group_id_mapping_file
                else:
                    # Try relative to data_root if available
                    data_root = kwargs.get('data_root', '')
                    if data_root:
                        group_id_mapping_path = Path(data_root) / group_id_mapping_file
            
            if group_id_mapping_path.exists():
                with open(group_id_mapping_path, 'r', encoding='utf-8') as f:
                    self.group_id_mapping = json.load(f)
                # Convert string keys to int
                self.group_id_mapping = {
                    int(k): v for k, v in self.group_id_mapping.items()
                }
                print(f'Loaded group_id mapping from {group_id_mapping_path}')
                print(f'Total mappings: {len(self.group_id_mapping)}')
                if self.group_id_weight:
                    print(f'Group ID weights: {self.group_id_weight}')
            else:
                print(f'Warning: group_id_mapping_file not found: {group_id_mapping_path}')
        
        super().__init__(ann_file=ann_file, **kwargs)
    
    def load_data_list(self) -> List[dict]:
        """Load data list from COCO annotation file with debugging."""
        from mmengine.dist import get_dist_info
        rank, world_size = get_dist_info()
        import logging
        logger = logging.getLogger('mmpose')
        
        logger.warning(
            f'[Rank {rank}/{world_size}] CocoParallelDataset.load_data_list: Starting to load data. '
            f'ann_file: {self.ann_file}, data_root: {self.data_root}'
        )
        
        # Call parent method
        data_list = super().load_data_list()
        
        logger.warning(
            f'[Rank {rank}/{world_size}] CocoParallelDataset.load_data_list: Loaded {len(data_list)} samples'
        )
        
        if len(data_list) == 0:
            logger.error(
                f'[Rank {rank}/{world_size}] ERROR: load_data_list returned empty list! '
                f'This may indicate that all instances were filtered out by parse_data_info.'
            )
        
        return data_list
    
    def _load_annotations(self):
        """Load data from annotations with debugging."""
        from mmengine.dist import get_dist_info
        rank, world_size = get_dist_info()
        import logging
        logger = logging.getLogger('mmpose')
        
        logger.warning(
            f'[Rank {rank}/{world_size}] CocoParallelDataset._load_annotations: Starting. '
            f'ann_file: {self.ann_file}'
        )
        
        # Call parent method
        instance_list, image_list = super()._load_annotations()
        
        logger.warning(
            f'[Rank {rank}/{world_size}] CocoParallelDataset._load_annotations: '
            f'Loaded {len(instance_list)} instances, {len(image_list)} images'
        )
        
        if len(instance_list) == 0:
            logger.error(
                f'[Rank {rank}/{world_size}] ERROR: instance_list is empty! '
                f'This means parse_data_info returned None for all annotations. '
                f'image_list size: {len(image_list)}'
            )
        
        return instance_list, image_list

    def parse_data_info(self, raw_data_info: dict):
        """Parse raw COCO annotation and add group_id information."""
        data_info = super().parse_data_info(raw_data_info)
        
        if data_info is None:
            return None
        
        # Try to get group_id from raw annotation first (if COCO file contains it)
        ann = raw_data_info.get('raw_ann_info', {})
        group_id = ann.get('group_id')
        
        # If not in raw annotation, try to get from mapping file
        if group_id is None:
            ann_id = data_info.get('id')
            if ann_id is not None and self.group_id_mapping:
                group_id = self.group_id_mapping.get(ann_id)
        
        # Set group_id and sample weight
        if group_id is not None:
            data_info['group_id'] = group_id
            
            # Add sample weight based on group_id
            if group_id in self.group_id_weight:
                weight = self.group_id_weight[group_id]
                data_info['sample_weight'] = weight
            else:
                data_info['sample_weight'] = 1.0
        else:
            data_info['group_id'] = None
            data_info['sample_weight'] = 1.0
        
        return data_info

    def _get_bottomup_data_infos(self, instance_list: List[Dict],
                                 image_list: List[Dict]) -> List[Dict]:
        """Organize the data list in bottom-up mode with sample weights."""
        # Debug: log instance_list size (log from all ranks for debugging)
        from mmengine.dist import get_dist_info
        rank, world_size = get_dist_info()
        import logging
        logger = logging.getLogger('mmpose')
        
        logger.warning(
            f'[Rank {rank}/{world_size}] CocoParallelDataset._get_bottomup_data_infos: '
            f'instance_list size: {len(instance_list)}, image_list size: {len(image_list)}'
        )
        
        if len(instance_list) == 0:
            logger.error(
                f'[Rank {rank}/{world_size}] ERROR: instance_list is empty! '
                f'This means parse_data_info returned None for all instances. '
                f'image_list size: {len(image_list)}'
            )
        
        # Call parent method first
        data_list_bu = super()._get_bottomup_data_infos(instance_list, image_list)
        
        # Debug: log data_list_bu size (log from all ranks)
        logger.warning(
            f'[Rank {rank}/{world_size}] CocoParallelDataset._get_bottomup_data_infos: '
            f'data_list_bu size after parent call: {len(data_list_bu)}'
        )
        
        if len(data_list_bu) == 0:
            logger.error(
                f'[Rank {rank}/{world_size}] ERROR: data_list_bu is empty after parent call! '
                f'instance_list size: {len(instance_list)}, image_list size: {len(image_list)}'
            )
        
        # Apply sample weights to instances in bottom-up mode
        # For bottom-up, we need to store sample_weight per instance
        # This will be used later in the pipeline to weight the loss
        for data_info_bu in data_list_bu:
            if 'id' in data_info_bu and isinstance(data_info_bu['id'], list):
                # Get group_ids and sample weights for each instance
                group_ids = []
                instance_weights = []
                for ann_id in data_info_bu['id']:
                    # Try to get group_id from mapping file first
                    group_id = None
                    if ann_id in self.group_id_mapping:
                        group_id = self.group_id_mapping[ann_id]
                    else:
                        # Try to get from instance_list (if parse_data_info already set it)
                        # Find the instance in instance_list
                        for inst in instance_list:
                            if inst.get('id') == ann_id:
                                group_id = inst.get('group_id')
                                break
                    
                    group_ids.append(group_id)
                    
                    # Calculate weight based on group_id
                    if group_id is not None and group_id in self.group_id_weight:
                        instance_weights.append(self.group_id_weight[group_id])
                    else:
                        instance_weights.append(1.0)
                
                # Store group_ids and instance weights (will be used in pipeline and sampler)
                data_info_bu['group_id'] = group_ids  # List of group_ids for WeightedGroupSampler
                data_info_bu['instance_weights'] = instance_weights
        
        return data_list_bu

