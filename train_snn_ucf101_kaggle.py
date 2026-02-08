"""
Training script for VGGSNN with TinyController on UCF101 Kaggle format.

Usage on Kaggle:
    !python train_snn_ucf101_kaggle.py --data_dir /kaggle/input/ucf101-action-recognition

Author: Auto-generated for SNN_Pruning project
"""

import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.functional import reset_net

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from archs.ucf101.VGGSNN_UCF101 import VGGSNN_UCF101
from datasets.ucf101_kaggle import get_ucf101_kaggle_loaders


def TET_loss(outputs, labels, criterion=nn.CrossEntropyLoss(), means=1.0, lamb=1e-3):
    """Temporal Efficient Training loss."""
    outputs = outputs.permute(1, 0, 2)  # [T, N, C] -> [N, T, C]
    T = outputs.size(1)
    
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T
    
    if lamb != 0:
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = nn.MSELoss()(outputs, y)
    else:
        Loss_mmd = 0
    
    return (1 - lamb) * Loss_es + lamb * Loss_mmd


def get_temperature(epoch, total_epochs, tau_start=1.0, tau_end=0.1):
    """Anneal Gumbel-Softmax temperature."""
    decay = (tau_end / tau_start) ** (epoch / max(total_epochs - 1, 1))
    return tau_start * decay


def train_epoch(epoch, train_loader, model, optimizer, scheduler, writer):
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    total_skip = 0.0
    
    for batch_idx, (videos, labels) in enumerate(train_loader):
        videos = videos.cuda()  # [N, T, C, H, W]
        labels = labels.cuda()
        
        optimizer.zero_grad()
        
        outputs, info = model(videos)
        
        cls_loss = TET_loss(outputs, labels)
        skip_loss = model.compute_skip_loss(info)
        loss = cls_loss + skip_loss
        
        loss.backward()
        optimizer.step()
        reset_net(model)
        
        # Stats
        total_loss += loss.item()
        pred = outputs.mean(0).argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        total_skip += info['skip_ratio'].item()
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Skip: {info['skip_ratio'].item():.2%}")
    
    if scheduler:
        scheduler.step()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    avg_skip = total_skip / len(train_loader)
    
    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/accuracy', accuracy, epoch)
    writer.add_scalar('train/skip_ratio', avg_skip, epoch)
    
    return avg_loss, accuracy, avg_skip


def evaluate(epoch, val_loader, model, writer):
    model.eval()
    
    correct = 0
    total = 0
    total_skip = 0.0
    
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.cuda()
            labels = labels.cuda()
            
            outputs, info = model(videos)
            
            pred = outputs.mean(0).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            total_skip += info['skip_ratio'].item()
            
            reset_net(model)
    
    accuracy = 100 * correct / total
    avg_skip = total_skip / len(val_loader)
    
    writer.add_scalar('val/accuracy', accuracy, epoch)
    writer.add_scalar('val/skip_ratio', avg_skip, epoch)
    
    return accuracy, avg_skip


def main():
    import argparse
    
    parser = argparse.ArgumentParser("SNN UCF101 Kaggle Training")
    parser.add_argument('--data_dir', type=str, 
                        default='/kaggle/input/ucf101-action-recognition',
                        help='Path to Kaggle UCF101 dataset')
    parser.add_argument('--frames_per_clip', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--target_skip', type=float, default=0.3)
    parser.add_argument('--frame_size', type=int, default=112)
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup
    save_dir = './snn_controller_ucf101'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter('./runs/ucf101_kaggle')
    
    print(f"Loading UCF101 from {args.data_dir}")
    
    # Load data
    train_loader, val_loader, num_classes = get_ucf101_kaggle_loaders(
        root=args.data_dir,
        frames_per_clip=args.frames_per_clip,
        batch_size=args.batch_size,
        frame_size=args.frame_size,
        num_workers=2
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    model = VGGSNN_UCF101(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        num_classes=num_classes,
        input_size=args.frame_size,
        controller_type='tiny',
        target_skip_ratio=args.target_skip
    ).cuda()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training
    best_accuracy = 0
    
    for epoch in range(1, args.epochs + 1):
        tau = get_temperature(epoch, args.epochs)
        model.set_controller_temperature(tau)
        
        print(f"\nEpoch {epoch}/{args.epochs} (tau={tau:.3f})")
        
        train_loss, train_acc, train_skip = train_epoch(
            epoch, train_loader, model, optimizer, scheduler, writer
        )
        
        val_acc, val_skip = evaluate(epoch, val_loader, model, writer)
        
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, Skip={train_skip:.2%}")
        print(f"  Val:   Acc={val_acc:.2f}%, Skip={val_skip:.2%}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'skip_ratio': val_skip,
            }, f"{save_dir}/best_model.pth")
            print(f"  -> New best: {val_acc:.2f}%")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{save_dir}/checkpoint_epoch{epoch}.pth")
    
    print(f"\n✅ Training complete! Best accuracy: {best_accuracy:.2f}%")
    writer.close()


if __name__ == '__main__':
    main()
