# Importing libraries
import os
import nibabel as nib
import torch
import glob
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random


# Definition of the custom Dataset
class LazyNii2DDataset(Dataset):
    def __init__(self, src_files, mask_files, target_shape):
        self.src_files = src_files
        self.mask_files = mask_files
        self.target_shape = target_shape

    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        # Load the source and mask file
        src_tensor = self.load_and_resize(self.src_files[idx])
        mask_tensor = self.load_and_resize(self.mask_files[idx])

        # Add the channel dimension
        src_tensor = src_tensor.unsqueeze(0)  # (1, H, W) for data
        mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W) for masks, if needed

        return src_tensor, mask_tensor

    def load_and_resize(self, file_path):
        nii = nib.load(file_path)
        img_data = nii.get_fdata()
        tensor_data = torch.tensor(img_data, dtype=torch.float32)

        # Add the channel dimension and resize to the target size
        tensor_data = tensor_data.unsqueeze(0)  # Add the channel dimension
        resized_tensor = F.interpolate(tensor_data.unsqueeze(0), size=self.target_shape, mode='bilinear', align_corners=False)
        return resized_tensor.squeeze()  # Remove the batch dimension

def dataLoaderMaking(namefile='CHAOS-MRT2-2D-NORMALIZED',target_shape = (256, 256),batch_size = 8):
    """
    Creates DataLoaders for training, validation, and testing from a given dataset directory. The data 
    is split into train, validation, and test sets, and corresponding DataLoaders are created for each set.

    Args:
        namefile (str): Directory containing the 2D source and mask `.nii.gz` files.
        target_shape (tuple): Target shape for resizing images and masks. Default is (256, 256).
        batch_size (int): Batch size for DataLoader. Default is 8.

    Returns:
        tuple: A tuple containing three DataLoaders:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
    """
    # Path to the folder containing the created 2D files
    chaos_dir = namefile

    # Retrieve all 2D source and mask files
    all_src_files = sorted(glob.glob(os.path.join(chaos_dir, '*-src-z*.nii.gz')))
    all_mask_files = sorted(glob.glob(os.path.join(chaos_dir, '*-mask-z*.nii.gz')))

    # Check that the number of source and mask files match
    assert len(all_src_files) == len(all_mask_files), "The number of source and mask files do not match!"

    # Group the files by identifier XX
    grouped_files = defaultdict(lambda: {'src': [], 'mask': []})

    for src_file, mask_file in zip(all_src_files, all_mask_files):
        identifier = src_file.split('/')[-1].split('-')[0]  # Extract XX from 'XX-T2SPIR-Y-z(number).nii.gz'
        grouped_files[identifier]['src'].append(src_file)
        grouped_files[identifier]['mask'].append(mask_file)

    # Convert the dictionary to lists for easier splitting
    group_keys = list(grouped_files.keys())
    random.seed(42)  # For reproducibility
    random.shuffle(group_keys)

    # Split the groups into train, val, and test sets
    train_split = int(0.8 * len(group_keys))  # 80% for training
    val_split = int(0.1 * len(group_keys))  # 10% for validation
    test_split = len(group_keys) - train_split - val_split  # Remaining 10% for testing

    train_keys = group_keys[:train_split]
    val_keys = group_keys[train_split:train_split + val_split]
    test_keys = group_keys[train_split + val_split:]

    # Function to gather files based on keys
    def get_files_from_keys(keys, grouped_files):
        """
        Prepares DataLoaders for training, validation, and testing by selecting files based on provided keys,
        creating datasets, and initializing DataLoaders.

        Args:
            keys (list): A list of keys corresponding to groups of source and mask file paths.
            grouped_files (dict): A dictionary where each key maps to a dictionary containing:
                - 'src' (list): Paths to source image files.
                - 'mask' (list): Paths to corresponding mask files.
            train_keys (list): Keys for selecting files for the training dataset.
            val_keys (list): Keys for selecting files for the validation dataset.
            test_keys (list): Keys for selecting files for the testing dataset.
            target_shape (tuple): Target shape to which all images and masks will be resized.
            batch_size (int): The batch size for the DataLoaders.

        Returns:
            tuple: A tuple containing three DataLoaders:
                - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
                - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
                - test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        """
        src_files = []
        mask_files = []
        for key in keys:
            src_files.extend(grouped_files[key]['src'])
            mask_files.extend(grouped_files[key]['mask'])
        return src_files, mask_files

    # Get the files for each set
    train_src, train_mask = get_files_from_keys(train_keys, grouped_files)
    val_src, val_mask = get_files_from_keys(val_keys, grouped_files)
    test_src, test_mask = get_files_from_keys(test_keys, grouped_files)

    # Create the datasets
    train_dataset = LazyNii2DDataset(train_src, train_mask, target_shape)
    val_dataset = LazyNii2DDataset(val_src, val_mask, target_shape)
    test_dataset = LazyNii2DDataset(test_src, test_mask, target_shape)

    # Create the DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,test_loader,val_loader
