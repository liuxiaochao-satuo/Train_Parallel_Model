# Copyright (c) OpenMMLab. All rights reserved.
import importlib.util
from pathlib import Path

from .builder import build_dataset
from .dataset_wrappers import CombinedDataset
from .datasets import *  # noqa

# MultiSourceSampler is in samplers.py (file in this directory)
# We need to import it directly from the file since samplers/ directory exists
_samplers_file = Path(__file__).parent / 'samplers.py'
if _samplers_file.exists():
    spec = importlib.util.spec_from_file_location('samplers_module', _samplers_file)
    samplers_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(samplers_module)
    MultiSourceSampler = samplers_module.MultiSourceSampler
else:
    from .samplers import MultiSourceSampler

# WeightedGroupSampler is in samplers/weighted_group_sampler.py (subdirectory)
from .samplers.weighted_group_sampler import WeightedGroupSampler
from .transforms import *  # noqa

__all__ = ['build_dataset', 'CombinedDataset', 'MultiSourceSampler', 'WeightedGroupSampler']
