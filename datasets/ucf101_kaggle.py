"""
Custom UCF101 Dataset Loader for Kaggle format.

Kaggle UCF101 format:
├── train/
│   ├── ApplyEyeMakeup/
│   ├── Archery/
│   └── ...
├── test/
├── val/
├── train.csv
├── test.csv
└── val.csv
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class UCF101Kaggle(Dataset):
    """
    UCF101 dataset loader for Kaggle format.
    
    Args:
        root: Path to dataset root (contains train/, test/, val/ folders)
        split: 'train', 'test', or 'val'
        frames_per_clip: Number of frames to sample per video
        transform: Optional transform for video frames
        frame_size: Size to resize frames (default: 112)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        frames_per_clip: int = 16,
        transform=None,
        frame_size: int = 112
    ):
        self.root = root
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.transform = transform
        self.frame_size = frame_size
        
        # Path to split folder
        self.split_dir = os.path.join(root, split)
        
        # Load CSV if exists, otherwise scan folders
        csv_path = os.path.join(root, f'{split}.csv')
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            self.use_csv = True
        else:
            self.use_csv = False
        
        # Get class names
        self.classes = sorted(os.listdir(self.split_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Build video list
        self.videos = []
        self._build_video_list()
        
        print(f"UCF101Kaggle: {split} split, {len(self.videos)} videos, {len(self.classes)} classes")
    
    def _build_video_list(self):
        """Build list of (video_path, label) tuples."""
        for class_name in self.classes:
            class_dir = os.path.join(self.split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            label = self.class_to_idx[class_name]
            
            for video_file in os.listdir(class_dir):
                if video_file.endswith(('.avi', '.mp4', '.mkv')):
                    video_path = os.path.join(class_dir, video_file)
                    self.videos.append((video_path, label))
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        Load video and sample frames uniformly.
        
        Returns:
            frames: Tensor of shape [T, C, H, W]
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            # Fallback: try to read all frames
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            total_frames = len(frames)
            cap.release()
            
            if total_frames == 0:
                # Return zeros if video is empty
                return torch.zeros(self.frames_per_clip, 3, self.frame_size, self.frame_size)
        else:
            frames = None
        
        # Sample frame indices uniformly
        if total_frames >= self.frames_per_clip:
            indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
        else:
            # Repeat frames if video is too short
            indices = np.linspace(0, total_frames - 1, self.frames_per_clip, dtype=int)
        
        # Read frames
        sampled_frames = []
        
        if frames is not None:
            # Already loaded
            for idx in indices:
                frame = frames[min(idx, len(frames)-1)]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                sampled_frames.append(frame)
        else:
            # Read from video
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                    sampled_frames.append(frame)
                else:
                    # Duplicate last frame if read fails
                    if sampled_frames:
                        sampled_frames.append(sampled_frames[-1].copy())
                    else:
                        sampled_frames.append(np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8))
            cap.release()
        
        # Convert to tensor [T, H, W, C] -> [T, C, H, W]
        frames_array = np.stack(sampled_frames, axis=0)
        frames_tensor = torch.from_numpy(frames_array).float() / 255.0
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        
        return frames_tensor
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path, label = self.videos[idx]
        
        # Load video frames
        frames = self._load_video(video_path)  # [T, C, H, W]
        
        # Apply transform if provided
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label


class VideoAugmentation:
    """Data augmentation for video."""
    
    def __init__(self, train=True):
        self.train = train
    
    def __call__(self, video):
        """
        Args:
            video: Tensor [T, C, H, W]
        Returns:
            Augmented video
        """
        if self.train:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                video = torch.flip(video, dims=[3])  # flip W dimension
            
            # Random brightness/contrast (slight)
            if torch.rand(1) > 0.5:
                brightness = 0.9 + 0.2 * torch.rand(1)
                video = video * brightness
        
        return video


def get_ucf101_kaggle_loaders(
    root: str,
    frames_per_clip: int = 16,
    batch_size: int = 8,
    frame_size: int = 112,
    num_workers: int = 4
):
    """
    Get train and validation data loaders for Kaggle UCF101.
    
    Args:
        root: Path to dataset root
        frames_per_clip: Number of frames per clip
        batch_size: Batch size
        frame_size: Frame size (112 or 224)
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, num_classes
    """
    train_dataset = UCF101Kaggle(
        root=root,
        split='train',
        frames_per_clip=frames_per_clip,
        transform=VideoAugmentation(train=True),
        frame_size=frame_size
    )
    
    val_dataset = UCF101Kaggle(
        root=root,
        split='val',
        frames_per_clip=frames_per_clip,
        transform=VideoAugmentation(train=False),
        frame_size=frame_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset.classes)


if __name__ == '__main__':
    # Test loading
    # Update this path to your Kaggle dataset path
    root = '/kaggle/input/ucf101-action-recognition'
    
    dataset = UCF101Kaggle(root, split='train', frames_per_clip=16)
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes[:10]}...")
    
    # Test loading one sample
    frames, label = dataset[0]
    print(f"Frames shape: {frames.shape}")  # Should be [16, 3, 112, 112]
    print(f"Label: {label}")
