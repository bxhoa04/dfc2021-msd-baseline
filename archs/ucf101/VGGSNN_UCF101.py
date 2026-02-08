"""
VGGSNN with Tiny Controller for UCF101 Action Recognition.

Adapted for RGB video input (3 channels) and 101 action classes.

Author: Auto-generated for SNN_Pruning project
"""

from typing import Any, Tuple
import torch
import torch.nn as nn
from copy import deepcopy

from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.layer import SeqToANNContainer
from spikingjelly.activation_based.functional import reset_net

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tiny_controller import TinyController, AdaptiveTimestepController


def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=bias)


class VGGSNN_UCF101(nn.Module):
    """
    VGG-style SNN for UCF101 action recognition with TinyController.
    
    Differences from CIFAR10-DVS version:
    - Input: 3 channels (RGB) instead of 2 (DVS events)
    - Output: 101 classes instead of 10
    - Larger spatial size handling (224x224 typical for video)
    
    Args:
        spiking_neuron: Spiking neuron class
        num_classes: Number of output classes (default: 101)
        input_size: Input spatial size (default: 112 for efficiency)
        controller_type: 'tiny' or 'adaptive'
        controller_hidden: Hidden dim for controller
        controller_tau: Gumbel-Softmax temperature
        skip_penalty: Weight for skip loss
        target_skip_ratio: Target skip ratio
    """
    
    def __init__(
        self,
        spiking_neuron: callable = None,
        num_classes: int = 101,
        input_size: int = 112,
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
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.controller_type = controller_type
        self.skip_penalty = skip_penalty
        self.target_skip_ratio = target_skip_ratio
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                SeqToANNContainer(
                    conv3x3(in_channels, out_channels, bias=False),
                    nn.BatchNorm2d(out_channels)
                ),
                spiking_neuron(**deepcopy(kwargs))
            )
        
        # Feature extractor - adapted for RGB input (3 channels)
        self.feature_extractor = nn.Sequential(
            conv_block(3, 64),      # Changed from 2 to 3 channels
            conv_block(64, 128),
            SeqToANNContainer(nn.AvgPool2d(2, 2)),  # 112 -> 56
            
            conv_block(128, 256),
            conv_block(256, 256),
            SeqToANNContainer(nn.AvgPool2d(2, 2)),  # 56 -> 28
            
            conv_block(256, 512),
            conv_block(512, 512),
            SeqToANNContainer(nn.AvgPool2d(2, 2)),  # 28 -> 14
            
            conv_block(512, 512),
            conv_block(512, 512),
            SeqToANNContainer(nn.AvgPool2d(2, 2))   # 14 -> 7
        )
        
        # Calculate feature size after convolutions
        # For input_size=112: 112 -> 56 -> 28 -> 14 -> 7
        feature_size = input_size // 16  # 4 pooling layers
        
        # Classifier - adapted for 101 classes
        self.classifier = nn.Sequential(
            SeqToANNContainer(nn.Flatten()),
            SeqToANNContainer(nn.Dropout(0.5)),
            SeqToANNContainer(nn.Linear(512 * feature_size * feature_size, 512, bias=False)),
            spiking_neuron(**deepcopy(kwargs)),
            SeqToANNContainer(nn.Dropout(0.25)),
            SeqToANNContainer(nn.Linear(512, num_classes, bias=False)),
        )
        
        # Tiny Controller
        if controller_type == 'tiny':
            self.controller = TinyController(
                input_dim=512,
                hidden_dim=controller_hidden,
                tau=controller_tau
            )
        else:
            self.controller = AdaptiveTimestepController(
                input_dim=512,
                hidden_dim=controller_hidden,
                tau=controller_tau
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def set_controller_temperature(self, tau: float):
        self.controller.set_temperature(tau)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with dynamic timestep skipping.
        
        Args:
            x: Input tensor [N, T, C, H, W] - RGB video frames
        
        Returns:
            outputs: Output tensor [T_processed, N, num_classes]
            info: Dictionary with skip statistics
        """
        N, T, C, H, W = x.shape
        device = x.device
        
        # Transpose: [N, T, C, H, W] -> [T, N, C, H, W]
        x = x.transpose(0, 1)
        
        outputs = []
        decisions = []
        process_probs = []
        h_controller = None
        last_features = None
        last_output = None
        
        for t in range(T):
            frame = x[t:t+1]  # [1, N, C, H, W]
            
            should_process = True
            if t > 0 and last_features is not None:
                features_for_ctrl = last_features.squeeze(0)
                
                if self.controller_type == 'tiny':
                    decision, gate = self.controller(features_for_ctrl)
                else:
                    decision, gate, h_controller = self.controller(
                        features_for_ctrl, h_controller
                    )
                
                decisions.append(decision)
                process_probs.append(gate[:, 1])
                
                if not self.training:
                    should_process = decision.float().mean() > 0.5
            
            if should_process:
                features = self.feature_extractor(frame)
                out = self.classifier(features)
                
                if t > 0 and self.training and len(process_probs) > 0:
                    gate_weight = process_probs[-1].unsqueeze(0).unsqueeze(-1)
                    out = out * gate_weight
                
                outputs.append(out)
                last_features = features
                last_output = out
            else:
                if last_output is not None:
                    outputs.append(last_output.clone())
        
        outputs = torch.cat(outputs, dim=0)
        
        if len(decisions) > 0:
            decisions_tensor = torch.stack(decisions, dim=0)
            process_probs_tensor = torch.stack(process_probs, dim=0)
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
        skip_ratio = info['skip_ratio']
        target = torch.tensor(self.target_skip_ratio, device=skip_ratio.device)
        return self.skip_penalty * (skip_ratio - target).pow(2)


if __name__ == '__main__':
    print("Testing VGGSNN_UCF101...")
    
    model = VGGSNN_UCF101(
        spiking_neuron=neuron.LIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        num_classes=101,
        input_size=112
    )
    
    # Test input: [N, T, C, H, W] = [2, 16, 3, 112, 112]
    x = torch.randn(2, 16, 3, 112, 112)
    
    model.eval()
    with torch.no_grad():
        outputs, info = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Skip ratio: {info['skip_ratio']:.2%}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
