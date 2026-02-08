"""
Training script for VGGSNN with TinyController.

This script trains the VGGSNN model with dynamic timestep skipping
controlled by a tiny neural network.

Usage:
    python train_snn_with_controller.py --dataset cifar10dvs --arch vggsnn_ctrl

Author: Auto-generated for SNN_Pruning project
"""

import os
import sys
import copy
import math
import random
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from spikingjelly.activation_based import neuron, surrogate, functional
from spikingjelly.activation_based.functional import reset_net
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

import utils
import config_lth

# Import model with controller
from archs.cifar10dvs.VGGSNN_with_controller import VGGSNN_WithController, TET_loss


# ==================== Data Augmentation ====================

class Augment:
    """Data augmentation for DVS data."""
    
    class Cutout:
        def __init__(self, ratio):
            self.ratio = ratio

        def __call__(self, img):
            h = img.size(1)
            w = img.size(2)
            lenth_h = int(self.ratio * h)
            lenth_w = int(self.ratio * w)
            mask = np.ones((h, w), np.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - lenth_h // 2, 0, h)
            y2 = np.clip(y + lenth_h // 2, 0, h)
            x1 = np.clip(x - lenth_w // 2, 0, w)
            x2 = np.clip(x + lenth_w // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
            return img

    class Roll:
        def __init__(self, off):
            self.off = off

        def __call__(self, img):
            off1 = random.randint(-self.off, self.off)
            off2 = random.randint(-self.off, self.off)
            return torch.roll(img, shifts=(off1, off2), dims=(1, 2))

    def function_nda(self, data, M=1, N=2):
        c = 15 * N
        rotate = transforms.RandomRotation(degrees=c)
        e = N / 6
        cutout = self.Cutout(ratio=e)
        a = N * 2 + 1
        roll = self.Roll(off=a)
        transforms_list = [roll, rotate, cutout]
        sampled_ops = np.random.choice(transforms_list, M)
        for op in sampled_ops:
            data = op(data)
        return data

    def __call__(self, img):
        flip = random.random() > 0.5
        if flip:
            img = torch.flip(img, dims=(2,))
        img = self.function_nda(img)
        return img


class DVStransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        shape = [img.shape[0], img.shape[1]]
        img = img.flatten(0, 1)
        img = self.transform(img)
        shape.extend(img.shape[1:])
        return img.view(shape)


class DatasetSplitter(torch.utils.data.Dataset):
    """Split CIFAR10DVS into training and test sets."""
    
    def __init__(self, parent_dataset, rate=0.1, train=True):
        self.parent_dataset = parent_dataset
        self.rate = rate
        self.train = train
        self.it_of_original = len(parent_dataset) // 10
        self.it_of_split = int(self.it_of_original * rate)

    def __len__(self):
        return int(len(self.parent_dataset) * self.rate)

    def __getitem__(self, index):
        base = (index // self.it_of_split) * self.it_of_original
        off = index % self.it_of_split
        if not self.train:
            off = self.it_of_original - off - 1
        item = self.parent_dataset[base + off]
        return item


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.dataset[index][0]), self.dataset[index][1]

    def __len__(self):
        return len(self.dataset)


def load_data(dataset_dir, T: int, distributed=False):
    """Load and prepare CIFAR10-DVS dataset."""
    
    transform_train = DVStransform(transform=transforms.Compose([
        transforms.Resize(size=(48, 48), antialias=True),
        Augment()
    ]))
    transform_test = DVStransform(
        transform=transforms.Resize(size=(48, 48), antialias=True)
    )
    
    dataset = CIFAR10DVS(dataset_dir, data_type='frame', frames_number=T, split_by='number')
    dataset_train_split = DatasetSplitter(dataset, 0.9, True)
    dataset_test_split = DatasetSplitter(dataset, 0.1, False)
    
    dataset_train = DatasetWrapper(dataset_train_split, transform_train)
    dataset_test = DatasetWrapper(dataset_test_split, transform_test)
    
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    
    return dataset_train, dataset_test, train_sampler, test_sampler


# ==================== Training Functions ====================

def train_epoch(
    args,
    epoch: int,
    train_loader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    writer=None
):
    """Train for one epoch with controller."""
    
    model.train()
    EPS = 1e-6
    
    train_loss = 0.0
    train_cls_loss = 0.0
    train_skip_loss = 0.0
    train_samples = 0
    total_skip_ratio = 0.0
    
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        imgs = imgs.cuda()
        labels = labels.cuda()
        
        # Forward pass with controller
        outputs, info = model(imgs)
        
        # Compute losses
        cls_loss = TET_loss(outputs, labels)
        skip_loss = model.compute_skip_loss(info)
        total_loss = cls_loss + skip_loss
        
        # Backward pass
        total_loss.backward()
        
        # Zero out gradients for pruned weights (if any)
        for name, p in model.named_parameters():
            if 'weight' in name and p.grad is not None:
                tensor = p.data
                if len(tensor.size()) == 1:
                    continue
                grad_tensor = p.grad
                grad_tensor = torch.where(
                    tensor.abs() < EPS, 
                    torch.zeros_like(grad_tensor), 
                    grad_tensor
                )
                p.grad.data = grad_tensor
        
        optimizer.step()
        reset_net(model)
        
        # Accumulate stats
        batch_size = labels.numel()
        train_samples += batch_size
        train_loss += total_loss.item() * batch_size
        train_cls_loss += cls_loss.item() * batch_size
        train_skip_loss += skip_loss.item() * batch_size
        total_skip_ratio += info['skip_ratio'].item() * batch_size
    
    # Average stats
    train_loss /= train_samples
    train_cls_loss /= train_samples
    train_skip_loss /= train_samples
    avg_skip_ratio = total_skip_ratio / train_samples
    
    if scheduler is not None:
        scheduler.step()
    
    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/cls_loss', train_cls_loss, epoch)
        writer.add_scalar('train/skip_loss', train_skip_loss, epoch)
        writer.add_scalar('train/skip_ratio', avg_skip_ratio, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
    
    return train_loss, avg_skip_ratio


def evaluate(model: nn.Module, test_loader, epoch: int = 0, writer=None):
    """Evaluate model on test set."""
    
    model.eval()
    test_samples = 0
    test_acc = 0
    total_skip_ratio = 0.0
    total_processed = 0
    total_timesteps = 0
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.cuda()
            label = label.cuda()
            
            outputs, info = model(data)
            
            # Mean over timesteps for prediction
            out_mean = outputs.mean(0)  # [N, C]
            
            test_samples += label.numel()
            test_acc += (out_mean.argmax(1) == label).float().sum().item()
            total_skip_ratio += info['skip_ratio'].item() * label.numel()
            total_processed += info['num_processed']
            total_timesteps += info['total_timesteps']
            
            reset_net(model)
    
    accuracy = 100.0 * test_acc / test_samples
    avg_skip_ratio = total_skip_ratio / test_samples
    avg_processed = total_processed / (test_samples / data.size(0))  # per sample
    
    if writer is not None:
        writer.add_scalar('test/accuracy', accuracy, epoch)
        writer.add_scalar('test/skip_ratio', avg_skip_ratio, epoch)
        writer.add_scalar('test/avg_processed_timesteps', avg_processed, epoch)
    
    return accuracy, avg_skip_ratio


# ==================== Temperature Annealing ====================

def get_temperature(epoch: int, total_epochs: int, tau_start: float = 1.0, tau_end: float = 0.1):
    """
    Anneal Gumbel-Softmax temperature from tau_start to tau_end.
    
    Uses exponential decay schedule.
    """
    decay = (tau_end / tau_start) ** (epoch / max(total_epochs - 1, 1))
    return tau_start * decay


# ==================== Main ====================

def main():
    args = config_lth.get_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup directories and logging
    exp_name = f"controller_VGGSNN/{args.dataset}_{args.arch}"
    writer = SummaryWriter(f"./runs/{exp_name}")
    save_dir = f"{os.getcwd()}/snn_controller/{args.arch}/{args.dataset}"
    utils.checkdir(save_dir)
    
    print(f"Experiment: {exp_name}")
    print(f"Save directory: {save_dir}")
    
    # Load data
    dataset_train, dataset_test, train_sampler, test_sampler = load_data(
        args.data_dir, args.timestep
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True, drop_last=False, num_workers=4
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler,
        pin_memory=True, drop_last=False, num_workers=4
    )
    
    # Create model with controller
    model = VGGSNN_WithController(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        controller_type='tiny',
        controller_hidden=32,
        controller_tau=1.0,
        skip_penalty=0.1,
        target_skip_ratio=0.3
    ).cuda()
    
    # Try to use cupy backend for speedup
    try:
        functional.set_backend(model, 'cupy')
        print("Using cupy backend")
    except:
        print("Cupy not available, using default backend")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    controller_params = sum(p.numel() for p in model.controller.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Controller parameters: {controller_params:,} ({100*controller_params/total_params:.2f}%)")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.end_iter, eta_min=0
    )
    
    # Training loop
    best_accuracy = 0
    tau_start = 1.0
    tau_end = 0.1
    
    for epoch in range(1, args.end_iter + 1):
        # Anneal temperature
        tau = get_temperature(epoch, args.end_iter, tau_start, tau_end)
        model.set_controller_temperature(tau)
        writer.add_scalar('controller/temperature', tau, epoch)
        
        # Train
        train_loss, train_skip_ratio = train_epoch(
            args, epoch, train_loader, model, optimizer, scheduler, writer
        )
        
        # Evaluate periodically
        if epoch % args.valid_freq == 0 or epoch == 1:
            accuracy, test_skip_ratio = evaluate(model, test_loader, epoch, writer)
            
            print(f"Epoch {epoch}/{args.end_iter} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Acc: {accuracy:.2f}% | "
                  f"Skip: {test_skip_ratio:.2%} | "
                  f"Tau: {tau:.3f}")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': accuracy,
                    'skip_ratio': test_skip_ratio,
                }, f"{save_dir}/best_model.pth")
                print(f"  -> New best accuracy: {accuracy:.2f}%")
        
        # Save checkpoint
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f"{save_dir}/checkpoint_epoch{epoch}.pth")
    
    # Final evaluation
    final_acc, final_skip = evaluate(model, test_loader, args.end_iter, writer)
    print(f"\nTraining complete!")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Final accuracy: {final_acc:.2f}%")
    print(f"Final skip ratio: {final_skip:.2%}")
    
    writer.close()


if __name__ == '__main__':
    main()
