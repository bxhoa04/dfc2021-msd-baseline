"""
VGGSNN with Tiny Controller for Dynamic Temporal Pruning.

This module extends VGGSNN with a TinyController that learns to skip
uninformative timesteps during inference, reducing computation while
maintaining accuracy.

Author: Auto-generated for SNN_Pruning project
"""

from typing import Any, Tuple, List
import torch
import torch.nn as nn
from copy import deepcopy

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.layer import SeqToANNContainer
from spikingjelly.activation_based.functional import reset_net

# Import the tiny controller
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiny_controller import TinyController, AdaptiveTimestepController


def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution layer"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=bias)


class VGGSNN_WithController(nn.Module):
    """
    VGG-style SNN with TinyController for dynamic timestep skipping.
    
    The controller observes the hidden state after feature extraction
    and decides whether to process or skip the next timestep.
    
    Args:
        spiking_neuron: Spiking neuron class (default: LIFNode)
        controller_type: 'tiny' or 'adaptive' (with GRU)
        controller_hidden: Hidden dimension for controller
        controller_tau: Initial Gumbel-Softmax temperature
        skip_penalty: Regularization weight for skip ratio
        target_skip_ratio: Target fraction of timesteps to skip
        **kwargs: Arguments passed to spiking neurons
    """
    
    def __init__(
        self,
        spiking_neuron: callable = None,
        controller_type: str = 'tiny',
        controller_hidden: int = 32,
        controller_tau: float = 1.0,
        skip_penalty: float = 0.1,
        target_skip_ratio: float = 0.3,
        **kwargs: Any
    ):
        super().__init__()
        
        if spiking_neuron is None:
            spiking_neuron = neuron.LIFNode
        
        self.controller_type = controller_type
        self.skip_penalty = skip_penalty
        self.target_skip_ratio = target_skip_ratio
        
        # Build conv blocks
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                SeqToANNContainer(
                    conv3x3(in_channels, out_channels, bias=False),
                    nn.BatchNorm2d(out_channels)
                ),
                spiking_neuron(**deepcopy(kwargs))
            )
        
        # Feature extractor (same as original VGGSNN)
        self.feature_extractor = nn.Sequential(
            conv_block(2, 64),
            conv_block(64, 128),
            SeqToANNContainer(nn.AvgPool2d(2, 2)),
            
            conv_block(128, 256),
            conv_block(256, 256),
            SeqToANNContainer(nn.AvgPool2d(2, 2)),
            
            conv_block(256, 512),
            conv_block(512, 512),
            SeqToANNContainer(nn.AvgPool2d(2, 2)),
            
            conv_block(512, 512),
            conv_block(512, 512),
            SeqToANNContainer(nn.AvgPool2d(2, 2))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            SeqToANNContainer(nn.Flatten()),
            SeqToANNContainer(nn.Dropout(0.25)),
            SeqToANNContainer(nn.Linear(512 * 3 * 3, 100, bias=False)),
            spiking_neuron(**deepcopy(kwargs)),
            SeqToANNContainer(nn.AvgPool1d(10, 10))  # Voting Layer
        )
        
        # Tiny Controller - monitors hidden state after feature extraction
        # Input dim = 512 (channels after last conv block)
        if controller_type == 'tiny':
            self.controller = TinyController(
                input_dim=512,
                hidden_dim=controller_hidden,
                tau=controller_tau
            )
        elif controller_type == 'adaptive':
            self.controller = AdaptiveTimestepController(
                input_dim=512,
                hidden_dim=controller_hidden,
                tau=controller_tau
            )
        else:
            raise ValueError(f"Unknown controller_type: {controller_type}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def set_controller_temperature(self, tau: float):
        """Set Gumbel-Softmax temperature for controller."""
        self.controller.set_temperature(tau)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with dynamic timestep skipping.
        
        Args:
            x: Input tensor [N, T, C, H, W]
        
        Returns:
            outputs: Output tensor [T_processed, N, num_classes]
            info: Dictionary with skip statistics
        """
        N, T, C, H, W = x.shape
        device = x.device
        
        # Transpose for processing: [N, T, C, H, W] -> [T, N, C, H, W]
        x = x.transpose(0, 1)
        
        outputs = []
        decisions = []
        process_probs = []
        
        # Track hidden state for controller
        h_controller = None  # For adaptive controller
        last_output = None
        last_features = None
        
        for t in range(T):
            frame = x[t:t+1]  # [1, N, C, H, W]
            
            # Skip decision (except for first timestep)
            should_process = True
            if t > 0 and last_features is not None:
                # Get controller decision based on last hidden state
                # last_features: [1, N, C', H', W'] -> [N, C', H', W']
                features_for_ctrl = last_features.squeeze(0)
                
                if self.controller_type == 'tiny':
                    decision, gate = self.controller(features_for_ctrl)
                else:
                    decision, gate, h_controller = self.controller(
                        features_for_ctrl, h_controller
                    )
                
                decisions.append(decision)
                process_probs.append(gate[:, 1])  # prob of process
                
                if not self.training:
                    # Hard decision during inference
                    should_process = decision.float().mean() > 0.5
                else:
                    # Soft decision during training (always process, but weight output)
                    should_process = True
            
            if should_process:
                # Process this timestep
                features = self.feature_extractor(frame)  # [1, N, C', H', W']
                out = self.classifier(features)  # [1, N, num_classes]
                
                # Apply soft gating during training
                if t > 0 and self.training and len(process_probs) > 0:
                    # Weight output by process probability
                    gate_weight = process_probs[-1].unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
                    out = out * gate_weight
                
                outputs.append(out)
                last_features = features
                last_output = out
            else:
                # Skip: reuse last output
                if last_output is not None:
                    outputs.append(last_output.clone())
        
        # Stack outputs: list of [1, N, C] -> [T_out, N, C]
        outputs = torch.cat(outputs, dim=0)
        
        # Compute skip statistics
        if len(decisions) > 0:
            decisions_tensor = torch.stack(decisions, dim=0)  # [T-1, N]
            process_probs_tensor = torch.stack(process_probs, dim=0)  # [T-1, N]
            
            # Skip ratio (1 - average process probability)
            skip_ratio = 1.0 - process_probs_tensor.mean()
            avg_decision = decisions_tensor.float().mean()
        else:
            skip_ratio = torch.tensor(0.0, device=device)
            avg_decision = torch.tensor(1.0, device=device)
        
        info = {
            'skip_ratio': skip_ratio,
            'avg_decision': avg_decision,
            'num_processed': len(outputs),
            'total_timesteps': T,
        }
        
        return outputs, info
    
    def compute_skip_loss(self, info: dict) -> torch.Tensor:
        """
        Compute regularization loss for skip ratio.
        
        Encourages the controller to skip close to target_skip_ratio.
        """
        skip_ratio = info['skip_ratio']
        target = torch.tensor(self.target_skip_ratio, device=skip_ratio.device)
        loss = self.skip_penalty * (skip_ratio - target).pow(2)
        return loss
    
    def forward_all_timesteps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass without skipping (for baseline comparison).
        
        Args:
            x: Input tensor [N, T, C, H, W]
        
        Returns:
            outputs: Output tensor [T, N, num_classes]
        """
        x = x.transpose(0, 1)  # [T, N, C, H, W]
        features = self.feature_extractor(x)
        outputs = self.classifier(features)
        return outputs


def TET_loss(outputs, labels, criterion=nn.CrossEntropyLoss(), means=1.0, lamb=1e-3):
    """
    Temporal Efficient Training loss.
    
    Args:
        outputs: [T, N, C] model outputs
        labels: [N] ground truth labels
        criterion: Base loss function
        means: Target mean for regularization
        lamb: Weight for MSE regularization
    """
    outputs = outputs.permute(1, 0, 2)  # [N, T, C]
    T = outputs.size(1)
    
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T
    
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)
    else:
        Loss_mmd = 0
    
    return (1 - lamb) * Loss_es + lamb * Loss_mmd


if __name__ == '__main__':
    from spikingjelly.activation_based import neuron, surrogate
    
    print("Testing VGGSNN_WithController...")
    
    # Create model
    model = VGGSNN_WithController(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        controller_type='tiny',
        controller_hidden=32,
        target_skip_ratio=0.3
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    controller_params = sum(p.numel() for p in model.controller.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Controller parameters: {controller_params:,} ({100*controller_params/total_params:.2f}%)")
    
    # Test forward pass
    x = torch.randn(2, 10, 2, 48, 48)  # [N, T, C, H, W]
    
    # Training mode
    model.train()
    outputs, info = model(x)
    print(f"\nTraining mode:")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Skip ratio: {info['skip_ratio']:.3f}")
    print(f"  Num processed: {info['num_processed']}/{info['total_timesteps']}")
    
    # Compute loss
    labels = torch.randint(0, 10, (2,))
    cls_loss = TET_loss(outputs, labels)
    skip_loss = model.compute_skip_loss(info)
    total_loss = cls_loss + skip_loss
    print(f"  Classification loss: {cls_loss:.4f}")
    print(f"  Skip loss: {skip_loss:.4f}")
    print(f"  Total loss: {total_loss:.4f}")
    
    # Test backward
    total_loss.backward()
    print("  Backward pass: OK")
    
    # Reset network
    reset_net(model)
    
    # Eval mode
    model.eval()
    with torch.no_grad():
        outputs, info = model(x)
    print(f"\nEval mode:")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Skip ratio: {info['skip_ratio']:.3f}")
    print(f"  Num processed: {info['num_processed']}/{info['total_timesteps']}")
    
    print("\nAll tests passed!")
