# training/loss.py
"""Standardized loss computation for FusionLLM.

Provides consistent loss computation across:
- Standard cross-entropy with proper normalization
- MTP (Multi-Token Prediction) auxiliary loss
- MoE load-balancing auxiliary loss
- Z-loss for softmax stabilization
- Label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class LossConfig:
    """Configuration for loss computation."""
    # Cross-entropy settings
    label_smoothing: float = 0.0          # Label smoothing factor
    z_loss_weight: float = 0.0            # Z-loss weight for softmax stabilization
    
    # MTP settings
    mtp_loss_weight: float = 0.0          # MTP auxiliary loss weight
    mtp_layers: int = 1                   # Number of MTP layers
    
    # MoE settings
    moe_balance_loss_weight: float = 0.01 # MoE load-balancing loss weight
    moe_z_loss_weight: float = 0.001      # MoE router z-loss weight
    
    # Softcap settings
    logit_softcap: float = 15.0           # Logit softcap value (0 to disable)
    
    # Normalization
    reduction: str = 'sum'                # Reduction method ('sum', 'mean', 'none')
    ignore_index: int = -100              # Ignore index for padding tokens


class StandardCrossEntropy(nn.Module):
    """Standard cross-entropy loss with proper normalization."""
    
    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        num_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute cross-entropy loss.
        
        Args:
            logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len)
            num_tokens: Optional count of valid tokens for normalization
            
        Returns:
            Dictionary with 'loss' and 'perplexity'
        """
        # Flatten for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        # Compute cross-entropy
        loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.config.ignore_index,
            label_smoothing=self.config.label_smoothing,
            reduction='sum'
        )
        
        # Normalize by number of tokens
        if num_tokens is None:
            # Count non-ignored tokens
            num_tokens = (labels_flat != self.config.ignore_index).sum().item()
        
        if num_tokens > 0:
            loss = loss / num_tokens
        
        # Compute perplexity
        perplexity = torch.exp(loss)
        
        return {
            'loss': loss,
            'perplexity': perplexity,
            'num_tokens': num_tokens,
        }
    
    def compute_z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute z-loss for softmax stabilization.
        
        z = log(sum(exp(logits)))^2
        
        Args:
            logits: (batch, seq_len, vocab_size)
            
        Returns:
            Scalar z-loss
        """
        if self.config.z_loss_weight <= 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Compute log-sum-exp
        log_z = torch.logsumexp(logits, dim=-1)
        
        # Compute z-loss
        z_loss = (log_z ** 2).mean()
        
        return self.config.z_loss_weight * z_loss


class MTPLoss(nn.Module):
    """Multi-Token Prediction auxiliary loss."""
    
    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()
    
    def forward(
        self,
        mtp_logits: list[torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute MTP auxiliary loss.
        
        Args:
            mtp_logits: List of (batch, seq_len, vocab_size) tensors for each MTP layer
            labels: (batch, seq_len) target labels
            
        Returns:
            Dictionary with 'mtp_loss' and individual layer losses
        """
        if self.config.mtp_loss_weight <= 0 or not mtp_logits:
            return {
                'mtp_loss': torch.tensor(0.0, device=labels.device),
                'mtp_layer_losses': [],
            }
        
        layer_losses = []
        for i, logits in enumerate(mtp_logits):
            # Shift logits and labels for next-token prediction
            # logits[:, :-1, :] predicts labels[:, 1:]
            shifted_logits = logits[:, :-1, :].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            
            # Compute cross-entropy for this layer
            batch_size, seq_len, vocab_size = shifted_logits.shape
            loss = F.cross_entropy(
                shifted_logits.view(-1, vocab_size),
                shifted_labels.view(-1),
                ignore_index=self.config.ignore_index,
                reduction='sum'
            )
            
            # Normalize by number of valid tokens
            num_tokens = (shifted_labels != self.config.ignore_index).sum().item()
            if num_tokens > 0:
                loss = loss / num_tokens
            
            layer_losses.append(loss)
        
        # Average across layers
        mtp_loss = torch.stack(layer_losses).mean()
        
        return {
            'mtp_loss': self.config.mtp_loss_weight * mtp_loss,
            'mtp_layer_losses': layer_losses,
        }


class MoELoadBalancingLoss(nn.Module):
    """MoE load-balancing auxiliary loss."""
    
    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()
    
    def forward(
        self,
        router_probs: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute MoE load-balancing loss.
        
        Args:
            router_probs: (batch, seq_len, n_experts) router probabilities
            expert_mask: (batch, seq_len, n_experts) binary mask of selected experts
            
        Returns:
            Dictionary with 'balance_loss', 'z_loss', and 'routing_stats'
        """
        if self.config.moe_balance_loss_weight <= 0:
            return {
                'balance_loss': torch.tensor(0.0, device=router_probs.device),
                'z_loss': torch.tensor(0.0, device=router_probs.device),
                'routing_stats': {},
            }
        
        # Compute load-balancing loss
        # f_i = fraction of tokens routed to expert i
        # P_i = average router probability for expert i
        # L_balance = N * sum(f_i * P_i)
        
        n_experts = router_probs.shape[-1]
        
        # Fraction of tokens routed to each expert
        f = expert_mask.float().mean(dim=[0, 1])  # (n_experts,)
        
        # Average router probability for each expert
        P = router_probs.mean(dim=[0, 1])  # (n_experts,)
        
        # Load-balancing loss
        balance_loss = n_experts * (f * P).sum()
        
        # Z-loss for router stability
        z_loss = torch.tensor(0.0, device=router_probs.device)
        if self.config.moe_z_loss_weight > 0:
            log_z = torch.logsumexp(router_probs, dim=-1)
            z_loss = self.config.moe_z_loss_weight * (log_z ** 2).mean()
        
        # Routing statistics
        routing_stats = {
            'load_balance': balance_loss.item(),
            'expert_load_std': f.std().item(),
            'router_entropy': -(P * torch.log(P + 1e-8)).sum().item(),
        }
        
        return {
            'balance_loss': self.config.moe_balance_loss_weight * balance_loss,
            'z_loss': z_loss,
            'routing_stats': routing_stats,
        }


class FusionLLMLoss(nn.Module):
    """Combined loss for FusionLLM training."""
    
    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()
        
        self.cross_entropy = StandardCrossEntropy(config)
        self.mtp_loss = MTPLoss(config)
        self.moe_loss = MoELoadBalancingLoss(config)
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mtp_logits: Optional[list[torch.Tensor]] = None,
        router_probs: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
        num_tokens: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            logits: (batch, seq_len, vocab_size) main model logits
            labels: (batch, seq_len) target labels
            mtp_logits: Optional list of MTP layer logits
            router_probs: Optional MoE router probabilities
            expert_mask: Optional MoE expert mask
            num_tokens: Optional count of valid tokens
            
        Returns:
            Dictionary with all loss components
        """
        result = {}
        
        # Main cross-entropy loss
        ce_result = self.cross_entropy(logits, labels, num_tokens)
        result['loss'] = ce_result['loss']
        result['perplexity'] = ce_result['perplexity']
        result['num_tokens'] = ce_result['num_tokens']
        
        # Z-loss
        z_loss = self.cross_entropy.compute_z_loss(logits)
        result['z_loss'] = z_loss
        result['loss'] = result['loss'] + z_loss
        
        # MTP loss
        if mtp_logits and self.config.mtp_loss_weight > 0:
            mtp_result = self.mtp_loss(mtp_logits, labels)
            result['mtp_loss'] = mtp_result['mtp_loss']
            result['mtp_layer_losses'] = mtp_result['mtp_layer_losses']
            result['loss'] = result['loss'] + mtp_result['mtp_loss']
        else:
            result['mtp_loss'] = torch.tensor(0.0, device=logits.device)
            result['mtp_layer_losses'] = []
        
        # MoE loss
        if router_probs is not None and expert_mask is not None and self.config.moe_balance_loss_weight > 0:
            moe_result = self.moe_loss(router_probs, expert_mask)
            result['balance_loss'] = moe_result['balance_loss']
            result['z_loss_moe'] = moe_result['z_loss']
            result['routing_stats'] = moe_result['routing_stats']
            result['loss'] = result['loss'] + moe_result['balance_loss'] + moe_result['z_loss']
        else:
            result['balance_loss'] = torch.tensor(0.0, device=logits.device)
            result['z_loss_moe'] = torch.tensor(0.0, device=logits.device)
            result['routing_stats'] = {}
        
        return result


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    config: Optional[LossConfig] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Convenience function for loss computation.
    
    Args:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len)
        config: Optional loss configuration
        **kwargs: Additional arguments for FusionLLMLoss
        
    Returns:
        Dictionary with loss components
    """
    loss_fn = FusionLLMLoss(config)
    return loss_fn(logits, labels, **kwargs)
