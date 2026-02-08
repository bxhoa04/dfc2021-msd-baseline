"""
Training script for VGGSNN with TinyController on UCF101 dataset.

Usage:
    python train_snn_ucf101.py --data_dir /path/to/ucf101 --annotation_path /path/to/annotations

Author: Auto-generated for SNN_Pruning project
"""

import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import UCF101

from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.functional import reset_net

import utils
import config_lth
from archs.ucf101.VGGSNN_UCF101 import VGGSNN_UCF101


# ==================== Data Loading ====================

class VideoTransform:
    """Transform for UCF101 video clips."""
    
    def __init__(self, size=112, train=True):
        self.size = size
        self.train = train
        
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((size + 16, size + 16)),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __call__(self, video):
        """
        Args:
            video: [T, H, W, C] uint8 tensor
        Returns:
            video: [T, C, H, W] float tensor, normalized
        """
        # Convert to float and scale to [0, 1]
        video = video.float() / 255.0
        
        # [T, H, W, C] -> [T, C, H, W]
        video = video.permute(0, 3, 1, 2)
        
        # Apply transforms frame by frame
        T = video.shape[0]
        frames = []
        for t in range(T):
            frame = video[t]
            if self.train:
                # Random crop same position for all frames
                if t == 0:
                    i, j, h, w = transforms.RandomCrop.get_params(
                        frame, output_size=(self.size, self.size)
                    )
                frame = transforms.functional.resized_crop(
                    frame, i, j, h, w, (self.size, self.size)
                )
            else:
                frame = transforms.functional.resize(frame, (self.size, self.size))
            
            # Normalize
            frame = transforms.functional.normalize(
                frame,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            frames.append(frame)
        
        return torch.stack(frames, dim=0)  # [T, C, H, W]


def collate_fn(batch):
    """Custom collate function to handle variable length videos."""
    videos = []
    labels = []
    
    for video, audio, label in batch:
        videos.append(video)
        labels.append(label)
    
    # Stack videos: [N, T, C, H, W]
    videos = torch.stack(videos, dim=0)
    labels = torch.tensor(labels)
    
    return videos, labels


def load_ucf101(data_dir, annotation_path, frames_per_clip=16, train=True):
    """
    Load UCF101 dataset.
    
    Args:
        data_dir: Path to UCF101 video folder
        annotation_path: Path to annotation folder with train/test splits
        frames_per_clip: Number of frames per video clip (timesteps)
        train: Whether to load train or test split
    
    Returns:
        dataset: UCF101 dataset
    """
    transform = VideoTransform(size=112, train=train)
    
    dataset = UCF101(
        root=data_dir,
        annotation_path=annotation_path,
        frames_per_clip=frames_per_clip,
        step_between_clips=frames_per_clip,  # Non-overlapping clips
        fold=1,
        train=train,
        transform=transform,
        output_format='THWC'
    )
    
    return dataset


# ==================== Loss Function ====================

def TET_loss(outputs, labels, criterion=nn.CrossEntropyLoss(), means=1.0, lamb=1e-3):
    """Temporal Efficient Training loss."""
    outputs = outputs.permute(1, 0, 2)  # [T, N, C] -> [N, T, C]
    T = outputs.size(1)
    
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T
    
    if lamb != 0:
        MMDLoss = nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)
    else:
        Loss_mmd = 0
    
    return (1 - lamb) * Loss_es + lamb * Loss_mmd


# ==================== Training ====================

def train_epoch(epoch, train_loader, model, optimizer, scheduler, writer):
    model.train()
    
    train_loss = 0.0
    train_samples = 0
    total_skip_ratio = 0.0
    correct = 0
    
    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos = videos.cuda()  # [N, T, C, H, W]
        labels = labels.cuda()
        
        optimizer.zero_grad()
        
        outputs, info = model(videos)
        
        cls_loss = TET_loss(outputs, labels)
        skip_loss = model.compute_skip_loss(info)
        total_loss = cls_loss + skip_loss
        
        total_loss.backward()
        optimizer.step()
        reset_net(model)
        
        # Stats
        batch_size = labels.size(0)
        train_samples += batch_size
        train_loss += total_loss.item() * batch_size
        total_skip_ratio += info['skip_ratio'].item() * batch_size
        
        # Accuracy
        pred = outputs.mean(0).argmax(1)
        correct += (pred == labels).sum().item()
        
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {total_loss.item():.4f}")
    
    if scheduler:
        scheduler.step()
    
    train_loss /= train_samples
    accuracy = 100.0 * correct / train_samples
    skip_ratio = total_skip_ratio / train_samples
    
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/accuracy', accuracy, epoch)
    writer.add_scalar('train/skip_ratio', skip_ratio, epoch)
    
    return train_loss, accuracy, skip_ratio


def evaluate(epoch, test_loader, model, writer):
    model.eval()
    
    test_samples = 0
    correct = 0
    total_skip_ratio = 0.0
    
    with torch.no_grad():
        for videos, labels in test_loader:
            videos = videos.cuda()
            labels = labels.cuda()
            
            outputs, info = model(videos)
            
            pred = outputs.mean(0).argmax(1)
            correct += (pred == labels).sum().item()
            test_samples += labels.size(0)
            total_skip_ratio += info['skip_ratio'].item() * labels.size(0)
            
            reset_net(model)
    
    accuracy = 100.0 * correct / test_samples
    skip_ratio = total_skip_ratio / test_samples
    
    writer.add_scalar('test/accuracy', accuracy, epoch)
    writer.add_scalar('test/skip_ratio', skip_ratio, epoch)
    
    return accuracy, skip_ratio


# ==================== Temperature Annealing ====================

def get_temperature(epoch, total_epochs, tau_start=1.0, tau_end=0.1):
    decay = (tau_end / tau_start) ** (epoch / max(total_epochs - 1, 1))
    return tau_start * decay


# ==================== Main ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser("SNN UCF101 Training")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to UCF101 video folder')
    parser.add_argument('--annotation_path', type=str, required=True,
                        help='Path to annotation folder')
    parser.add_argument('--frames_per_clip', type=int, default=16,
                        help='Number of frames per clip (timesteps)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (smaller due to video size)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--target_skip', type=float, default=0.3,
                        help='Target skip ratio')
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    # Setup
    save_dir = f"{os.getcwd()}/snn_controller_ucf101"
    utils.checkdir(save_dir)
    writer = SummaryWriter(f"./runs/controller_UCF101")
    
    print(f"Loading UCF101 from {args.data_dir}...")
    print(f"Annotations: {args.annotation_path}")
    
    # Load data
    train_dataset = load_ucf101(
        args.data_dir, args.annotation_path,
        frames_per_clip=args.frames_per_clip, train=True
    )
    test_dataset = load_ucf101(
        args.data_dir, args.annotation_path,
        frames_per_clip=args.frames_per_clip, train=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = VGGSNN_UCF101(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        num_classes=101,
        input_size=112,
        controller_type='tiny',
        target_skip_ratio=args.target_skip
    ).cuda()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training loop
    best_accuracy = 0
    
    for epoch in range(1, args.epochs + 1):
        # Anneal temperature
        tau = get_temperature(epoch, args.epochs)
        model.set_controller_temperature(tau)
        
        print(f"\nEpoch {epoch}/{args.epochs} (tau={tau:.3f})")
        
        train_loss, train_acc, train_skip = train_epoch(
            epoch, train_loader, model, optimizer, scheduler, writer
        )
        
        test_acc, test_skip = evaluate(epoch, test_loader, model, writer)
        
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, Skip={train_skip:.2%}")
        print(f"  Test:  Acc={test_acc:.2f}%, Skip={test_skip:.2%}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': test_acc,
                'skip_ratio': test_skip,
            }, f"{save_dir}/best_model.pth")
            print(f"  -> New best: {test_acc:.2f}%")
    
    print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
    writer.close()


if __name__ == '__main__':
    main()
