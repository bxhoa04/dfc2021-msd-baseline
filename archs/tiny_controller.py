"""
Tiny Controller for Dynamic Temporal Pruning in SNN.

A lightweight neural network that predicts whether to process or skip
each timestep based on the current hidden state. Uses Gumbel-Softmax
for differentiable discrete decisions during training.

Author: Auto-generated for SNN_Pruning project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = False, dim: int = -1) -> torch.Tensor:
    """
    Gumbel-Softmax sampling for differentiable discrete decisions.
    
    Args:
        logits: Unnormalized log probabilities [N, num_classes]
        tau: Temperature parameter (lower = more discrete)
        hard: If True, returns one-hot vectors (straight-through estimator)
        dim: Dimension to apply softmax
    
    Returns:
        Sampled probabilities or one-hot vectors [N, num_classes]
    """
    # Sample from Gumbel(0, 1)
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.softmax(dim)
    
    if hard:
        # Straight-through estimator
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    
    return y_soft


class TinyController(nn.Module):
    """
    Tiny neural network to predict skip/process decision for each timestep.
    
    Architecture:
        - Global Average Pooling (reduces spatial dimensions)
        - Linear(input_dim, hidden_dim)
        - ReLU activation
        - Linear(hidden_dim, 2)  # 2 classes: skip=0, process=1
        - Gumbel-Softmax for differentiable sampling
    
    Args:
        input_dim: Number of input channels (from hidden state)
        hidden_dim: Hidden layer dimension (default: 32)
        tau: Gumbel-Softmax temperature (default: 1.0)
    
    Input:
        hidden_state: [N, C, H, W] tensor from previous layer
        
    Output:
        decision: [N] tensor of 0 (skip) or 1 (process)
        gate: [N, 2] soft probabilities for skip/process
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, tau: float = 1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        
        # Global average pooling is applied in forward()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 2 classes: skip, process
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights to start near 50-50 decision."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def set_temperature(self, tau: float):
        """Set Gumbel-Softmax temperature (for annealing during training)."""
        self.tau = tau
    
    def forward(self, hidden_state: torch.Tensor, hard: bool = None) -> tuple:
        """
        Forward pass to predict skip/process decision.
        
        Args:
            hidden_state: [N, C, H, W] tensor from backbone network
            hard: Override for hard decision (default: not training)
        
        Returns:
            decision: [N] tensor of 0 (skip) or 1 (process)
            gate: [N, 2] soft probabilities [p_skip, p_process]
        """
        # Global average pooling: [N, C, H, W] -> [N, C]
        x = hidden_state.mean(dim=[-2, -1])
        
        # MLP
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # [N, 2]
        
        # Gumbel-Softmax sampling
        if hard is None:
            hard = not self.training
        
        gate = gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)
        
        # Decision: 0=skip, 1=process (index of max)
        decision = gate[:, 1]  # probability of process
        
        if hard or not self.training:
            decision = (decision > 0.5).long()
        
        return decision, gate
    
    def get_process_probability(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Get soft probability of processing (for visualization/analysis)."""
        x = hidden_state.mean(dim=[-2, -1])
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]  # probability of process
    
    def extra_repr(self) -> str:
        return f'input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, tau={self.tau}'


class AdaptiveTimestepController(nn.Module):
    """
    Extended controller that considers temporal context.
    
    Uses a small GRU to maintain temporal state across timesteps,
    allowing more informed skip decisions based on history.
    
    Args:
        input_dim: Number of input channels
        hidden_dim: Hidden dimension (default: 32)
        tau: Gumbel-Softmax temperature
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, tau: float = 1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau = tau
        
        # Reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # GRU for temporal context
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        
        # Decision head
        self.fc = nn.Linear(hidden_dim, 2)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def set_temperature(self, tau: float):
        self.tau = tau
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize GRU hidden state."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)
    
    def forward(self, hidden_state: torch.Tensor, h: torch.Tensor = None, hard: bool = None) -> tuple:
        """
        Forward pass with temporal context.
        
        Args:
            hidden_state: [N, C, H, W] tensor from backbone
            h: [N, hidden_dim] GRU hidden state (optional)
            hard: Override for hard decision
        
        Returns:
            decision: [N] tensor of 0 (skip) or 1 (process)
            gate: [N, 2] soft probabilities
            h_new: [N, hidden_dim] updated hidden state
        """
        batch_size = hidden_state.size(0)
        device = hidden_state.device
        
        # Initialize hidden state if not provided
        if h is None:
            h = self.init_hidden(batch_size, device)
        
        # Pool spatial dimensions: [N, C, H, W] -> [N, C]
        x = self.pool(hidden_state).view(batch_size, -1)
        
        # Update temporal context
        h_new = self.gru(x, h)
        
        # Decision
        logits = self.fc(h_new)
        
        if hard is None:
            hard = not self.training
        
        gate = gumbel_softmax(logits, tau=self.tau, hard=hard, dim=-1)
        decision = gate[:, 1]
        
        if hard or not self.training:
            decision = (decision > 0.5).long()
        
        return decision, gate, h_new


if __name__ == '__main__':
    # Test TinyController
    print("Testing TinyController...")
    controller = TinyController(input_dim=512, hidden_dim=32)
    x = torch.randn(4, 512, 3, 3)  # [N, C, H, W]
    
    # Training mode
    controller.train()
    decision, gate = controller(x)
    print(f"Training - Decision shape: {decision.shape}, Gate shape: {gate.shape}")
    print(f"  Decision values: {decision}")
    print(f"  Process probs: {gate[:, 1]}")
    
    # Eval mode
    controller.eval()
    decision, gate = controller(x)
    print(f"Eval - Decision: {decision}")
    
    # Count parameters
    num_params = sum(p.numel() for p in controller.parameters())
    print(f"Number of parameters: {num_params}")
    
    # Test AdaptiveTimestepController
    print("\nTesting AdaptiveTimestepController...")
    adaptive_ctrl = AdaptiveTimestepController(input_dim=512, hidden_dim=32)
    h = None
    for t in range(5):
        x = torch.randn(4, 512, 3, 3)
        decision, gate, h = adaptive_ctrl(x, h)
        print(f"  t={t}: decision={decision.tolist()}")
