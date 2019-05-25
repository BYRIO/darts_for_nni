"""
File: base_dataloader.py
Author: VDeamoV
Email: vincent.duan95@outlook.com
Github: https://github.com/VDeamoV
Description:
    Create some special dataloader such as meta dataloader
"""

from torch.utils.data.dataloader import DataLoader

from . import sampler

class CustomMetaDataloader(DataLoader):
    """
    CustomMetaDataLoader inherit from torch.utils.data.dataloader,
    Simplely add basic Sampler needed for MetaLearing
    """
    pass
