"""
Dataset loaders and data utilities for CGL-RPCANet.
GSFANet-style data preprocessing and augmentation.
"""

import torch
import torch.utils.data as Data
import os
from PIL import Image
from scipy.ndimage import rotate
import os.path as osp
import random
import numpy as np


def get_img_norm_cfg(dataset_name, dataset_dir=None):
    """
    Get dataset-specific image normalization parameters (GSFANet-style).

    Args:
        dataset_name: Name of the dataset
        dataset_dir: Dataset directory (used for computing stats if unknown dataset)

    Returns:
        Dict with 'mean' and 'std' keys (in [0, 255] range)
    """
    if dataset_name == 'NUAA-SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1k':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'SIRST2':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST3':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'NUDT-SIRST-Sea-Light':
        img_norm_cfg = dict(mean=21.39715576171875, std=10.919337272644043)
    elif dataset_name == 'Maritime_sirst':
        img_norm_cfg = dict(mean=36.6230583190918, std=13.484057426452637)
    elif dataset_name == 'SIRST4':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST5':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST6':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST7':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'IRDST-real':
        img_norm_cfg = {'mean': 101.54053497314453, 'std': 56.49856185913086}
    else:
        # If unknown dataset and dataset_dir provided, compute stats
        if dataset_dir is not None and os.path.exists(dataset_dir):
            try:
                img_list = []
                # Try to find trainval.txt and test.txt
                trainval_file = osp.join(dataset_dir, 'trainval.txt')
                test_file = osp.join(dataset_dir, 'test.txt')
                
                if osp.exists(trainval_file) and osp.exists(test_file):
                    with open(trainval_file, 'r') as f:
                        img_list += f.read().splitlines()
                    with open(test_file, 'r') as f:
                        img_list += f.read().splitlines()
                    
                    img_dir = osp.join(dataset_dir, 'images')
                    mean_list = []
                    std_list = []
                    for img_pth in img_list:
                        try:
                            img = Image.open(osp.join(img_dir, img_pth + '.png')).convert('I')
                        except:
                            try:
                                img = Image.open(osp.join(img_dir, img_pth + '.jpg')).convert('I')
                            except:
                                img = Image.open(osp.join(img_dir, img_pth + '.bmp')).convert('I')
                        img = np.array(img, dtype=np.float32)
                        mean_list.append(img.mean())
                        std_list.append(img.std())
                    img_norm_cfg = dict(mean=float(np.array(mean_list).mean()), 
                                       std=float(np.array(std_list).mean()))
                    return img_norm_cfg
            except Exception as e:
                print(f"Warning: Failed to compute normalization stats: {e}")
        
        # Default fallback
        print(f"Warning: Unknown dataset '{dataset_name}', using default normalization")
        img_norm_cfg = dict(mean=100.0, std=35.0)
    
    return img_norm_cfg


def PadImg(img, times=32):
    """
    Pad image to be divisible by `times` (GSFANet-style).

    Args:
        img: Input image [H, W]
        times: Divisibility factor (default: 32)

    Returns:
        Padded image
    """
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img


class TrainDataset(Data.Dataset):
    """
    GSFANet-style dataset loader for infrared small target detection.
    
    Training mode:
        - Normalize with dataset-specific mean/std
        - Random crop with positive sample probability
        - Synchronized augmentation (flip, transpose, rotation)
    
    Validation mode:
        - Normalize with dataset-specific mean/std
        - Pad to be divisible by 32
    
    Args:
        args: Argument object with:
            - dataset_root: Root directory containing datasets
            - dataset: Dataset name (e.g., 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1k')
            - crop_size: Crop size for training
            - base_size: Base image size (not used in GSFANet-style)
        mode: 'train' or 'val'
        img_norm_cfg: Optional normalization config override
    """
    
    def __init__(self, args, mode='train', img_norm_cfg=None):
        self.args = args
        dataset_dir = args.dataset_root + '/' + args.dataset

        if args.dataset in ['Maritime_sirst', 'dataset']:
            if mode == 'train':
                self.imgs_dir = osp.join(dataset_dir, 'trainval', 'images')
                self.label_dir = osp.join(dataset_dir, 'trainval', 'masks')
            else:
                self.imgs_dir = osp.join(dataset_dir, 'detect', 'images')
                self.label_dir = osp.join(dataset_dir, 'detect', 'masks')

            self.names = []
            for filename in os.listdir(self.imgs_dir):
                if filename.endswith('png'):
                    base_name, _ = osp.splitext(filename)
                    self.names.append(base_name)
        else:
            if mode == 'train':
                txtfile = 'trainval.txt'
            elif mode == 'val' or mode == 'vis':
                txtfile = 'test.txt'

            self.list_dir = osp.join(dataset_dir, txtfile)
            self.imgs_dir = osp.join(dataset_dir, 'images')
            self.label_dir = osp.join(dataset_dir, 'masks')

            self.names = []
            with open(self.list_dir, 'r') as f:
                self.names += [line.strip() for line in f.readlines()]

        if img_norm_cfg is None:
            self.img_norm_cfg = get_img_norm_cfg(args.dataset, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name + '.png')
        if self.args.dataset == 'NUAA-SIRST':
            label_path = osp.join(self.label_dir, name + '_pixels0.png')
        else:
            label_path = osp.join(self.label_dir, name + '.png')

        img = Image.open(img_path).convert('L')
        mask = Image.open(label_path)

        if self.mode == 'train':
            img = (np.array(img, dtype=np.float32) - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
            mask = np.array(mask, dtype=np.float32) / 255.0

            img, mask = self._random_crop(np.array(img), np.array(mask), self.crop_size, pos_prob=0.5)
            img, mask = self._sync_transform(img, mask)

            img_batch, mask_batch = img[np.newaxis, :], mask[np.newaxis, :]
            img_batch = torch.from_numpy(np.ascontiguousarray(img_batch)).to(torch.float32)
            mask_batch = torch.from_numpy(np.ascontiguousarray(mask_batch)).to(torch.float32)
            return img_batch, mask_batch

        elif self.mode == 'val':
            img = (np.array(img, dtype=np.float32) - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
            mask = np.array(mask, dtype=np.float32) / 255.0
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]

            img = PadImg(img)
            mask = PadImg(mask)

            img, mask = img[np.newaxis, :], mask[np.newaxis, :]

            img = torch.from_numpy(np.ascontiguousarray(img)).to(torch.float32)
            mask = torch.from_numpy(np.ascontiguousarray(mask)).to(torch.float32)
            if img.shape != mask.shape:
                print('img!=mask in dataset')
            return img, mask

        else:
            raise ValueError("Unknown self.mode")

    def __len__(self):
        return len(self.names)

    def _random_crop(self, img, mask, patch_size, pos_prob=None):
        """
        Random crop with probabilistic positive sample guarantee (GSFANet-style).

        Strategy:
            - If image smaller than patch_size, pad with zeros
            - Loop until valid crop found:
              - Random crop position
              - If pos_prob is None OR random() > pos_prob: accept any crop
              - Else if crop contains target: accept
              - Otherwise: retry
        """
        h, w = img.shape
        if min(h, w) < patch_size:
            img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                         mode='constant')
            mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                          mode='constant')
            h, w = img.shape

        while 1:
            h_start = random.randint(0, h - patch_size)
            h_end = h_start + patch_size
            w_start = random.randint(0, w - patch_size)
            w_end = w_start + patch_size

            img_patch = img[h_start:h_end, w_start:w_end]
            mask_patch = mask[h_start:h_end, w_start:w_end]

            if pos_prob is None or random.random() > pos_prob:
                break
            elif mask_patch.sum() > 0:
                break

        return img_patch, mask_patch

    def _sync_transform(self, img, mask):
        """
        Apply GSFANet-style synchronized augmentations to image and mask.

        Augmentations:
        - Horizontal flip (50% prob)
        - Vertical flip (50% prob)
        - Transpose (50% prob)
        - Small rotation ±3° (50% prob)
        """
        # random mirror
        if random.random() < 0.5:  # 水平反转
            img = img[::-1, :]
            mask = mask[::-1, :]
        if random.random() < 0.5:  # 垂直反转
            img = img[:, ::-1]
            mask = mask[:, ::-1]
        if random.random() < 0.5:  # 转置反转
            img = img.transpose(1, 0)
            mask = mask.transpose(1, 0)
        # random rotate
        if random.random() < 0.5:
            angle = random.uniform(-3, 3)
            img = rotate(img, angle, reshape=False, order=1)  # bilinear
            mask = rotate(mask, angle, reshape=False, order=0)  # nearest

        return img, mask


# ==================== Legacy compatibility layer ====================
# These classes maintain backward compatibility with existing code

class ISTDDataset(Data.Dataset):
    """
    Legacy compatibility wrapper for ISTDDataset.
    Converts the old API to the new GSFANet-style TrainDataset.
    
    Args:
        root: Root directory path
        split: 'train' or 'val'
        split_file: Optional text file with image names (not used in GSFANet-style)
        image_dir: Subdirectory for images (default: 'images')
        mask_dir: Subdirectory for masks (default: 'masks')
        image_size: Base image size (not used in GSFANet-style)
        use_augmentation: Whether to apply augmentations (default: True for train)
        crop_size: Crop size for training (H, W) or int
        dataset_name: Dataset name for normalization config (default: 'NUAA-SIRST')
        img_norm_cfg: Optional override for normalization config
    """
    
    def __init__(
            self,
            root: str,
            split: str = 'train',
            split_file: str = None,
            image_dir: str = 'images',
            mask_dir: str = 'masks',
            image_size=None,
            use_augmentation: bool = None,
            crop_size=None,
            dataset_name: str = 'NUAA-SIRST',
            img_norm_cfg=None
    ):
        from pathlib import Path
        
        self.root = Path(root)
        self.split = split
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir
        self.dataset_name = dataset_name
        self.use_augmentation = use_augmentation if use_augmentation is not None else (split == 'train')
        
        # Crop size (convert tuple to int if needed)
        if crop_size is not None:
            self.crop_size = crop_size[0] if isinstance(crop_size, tuple) else crop_size
        else:
            self.crop_size = 256  # default

        # Get normalization config (GSFANet-style)
        if img_norm_cfg is not None:
            self.img_norm_cfg = img_norm_cfg
        else:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name)

        # Load file list
        if split_file is not None and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                self.image_names = [line.strip() for line in f if line.strip()]
        else:
            # All images in directory (get base names without extension)
            self.image_names = []
            for f in sorted(self.image_dir.glob('*.png')):
                self.image_names.append(f.stem)  # stem = filename without extension
            if len(self.image_names) == 0:
                for f in sorted(self.image_dir.glob('*.jpg')):
                    self.image_names.append(f.stem)

        assert len(self.image_names) > 0, f"No images found in {self.image_dir}"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Load one image-mask pair with GSFANet-style preprocessing.
        """
        name = self.image_names[idx]

        # Image path
        img_path = self.image_dir / f"{name}.png"
        if not img_path.exists():
            img_path = self.image_dir / f"{name}.jpg"

        # Mask path (NUAA-SIRST uses special naming)
        if self.dataset_name == 'NUAA-SIRST':
            mask_path = self.mask_dir / f"{name}_pixels0.png"
            if not mask_path.exists():
                mask_path = self.mask_dir / f"{name}.png"
        else:
            mask_path = self.mask_dir / f"{name}.png"

        # Load as grayscale
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.split == 'train':
            # Training: normalize -> crop -> augment
            img_np = (np.array(img, dtype=np.float32) - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
            mask_np = np.array(mask, dtype=np.float32) / 255.0

            # Ensure mask is 2D
            if len(mask_np.shape) > 2:
                mask_np = mask_np[:, :, 0]

            # Random crop (GSFANet-style with pos_prob=0.5)
            if self.crop_size is not None:
                img_np, mask_np = self._random_crop(img_np, mask_np, self.crop_size, pos_prob=0.5)

            # Synchronized augmentation
            if self.use_augmentation:
                img_np, mask_np = self._sync_transform(img_np, mask_np)

            # To tensor [1, H, W]
            img_batch = img_np[np.newaxis, :]
            mask_batch = mask_np[np.newaxis, :]
            img_batch = torch.from_numpy(np.ascontiguousarray(img_batch)).to(torch.float32)
            mask_batch = torch.from_numpy(np.ascontiguousarray(mask_batch)).to(torch.float32)

            return img_batch, mask_batch

        else:
            # Validation: normalize -> pad to 32x (GSFANet-style)
            img_np = (np.array(img, dtype=np.float32) - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
            mask_np = np.array(mask, dtype=np.float32) / 255.0

            # Ensure mask is 2D
            if len(mask_np.shape) > 2:
                mask_np = mask_np[:, :, 0]

            # Pad to be divisible by 32 (GSFANet-style)
            img_np = PadImg(img_np)
            mask_np = PadImg(mask_np)

            # To tensor [1, H, W]
            img_batch = img_np[np.newaxis, :]
            mask_batch = mask_np[np.newaxis, :]
            img_batch = torch.from_numpy(np.ascontiguousarray(img_batch)).to(torch.float32)
            mask_batch = torch.from_numpy(np.ascontiguousarray(mask_batch)).to(torch.float32)

            return img_batch, mask_batch

    def _random_crop(self, img, mask, patch_size, pos_prob=None):
        """Random crop with positive sample probability (GSFANet-style)."""
        h, w = img.shape
        if min(h, w) < patch_size:
            img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                         mode='constant')
            mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                          mode='constant')
            h, w = img.shape

        while 1:
            h_start = random.randint(0, h - patch_size)
            h_end = h_start + patch_size
            w_start = random.randint(0, w - patch_size)
            w_end = w_start + patch_size

            img_patch = img[h_start:h_end, w_start:w_end]
            mask_patch = mask[h_start:h_end, w_start:w_end]

            if pos_prob is None or random.random() > pos_prob:
                break
            elif mask_patch.sum() > 0:
                break

        return img_patch, mask_patch

    def _sync_transform(self, img, mask):
        """Apply GSFANet-style synchronized augmentations."""
        # random mirror
        if random.random() < 0.5:  # 水平反转
            img = img[::-1, :]
            mask = mask[::-1, :]
        if random.random() < 0.5:  # 垂直反转
            img = img[:, ::-1]
            mask = mask[:, ::-1]
        if random.random() < 0.5:  # 转置反转
            img = img.transpose(1, 0)
            mask = mask.transpose(1, 0)
        # random rotate
        if random.random() < 0.5:
            angle = random.uniform(-3, 3)
            img = rotate(img, angle, reshape=False, order=1)  # bilinear
            mask = rotate(mask, angle, reshape=False, order=0)  # nearest

        return img, mask


def get_dataloader(
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
):
    """
    Create DataLoader with standard settings.

    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle (default: True)
        num_workers: Number of worker processes (default: 4)
        pin_memory: Pin memory for faster GPU transfer (default: True)

    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
