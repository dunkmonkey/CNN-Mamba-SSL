"""
InfoNCE Contrastive Loss for MoCo.

Implements the normalized temperature-scaled cross entropy loss
used in momentum contrast learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Normalized Temperature-scaled Cross Entropy) Loss.
    
    Given a query and a set of keys (one positive, rest negative),
    the loss encourages the query to be similar to its positive key
    and dissimilar to negative keys.
    
    Loss = -log(exp(q·k+ / τ) / Σ exp(q·ki / τ))
    
    where:
    - q: query embedding
    - k+: positive key embedding
    - ki: all key embeddings (positive + negatives)
    - τ: temperature parameter
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for scaling similarities.
                        Lower values make the loss more discriminative.
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: Query embeddings of shape (batch_size, dim)
            positive_key: Positive key embeddings of shape (batch_size, dim)
            negative_keys: Negative key embeddings of shape (queue_size, dim)
            
        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        query = F.normalize(query, dim=1)
        positive_key = F.normalize(positive_key, dim=1)
        negative_keys = F.normalize(negative_keys, dim=1)
        
        # Compute positive logits: (batch_size, 1)
        l_pos = torch.einsum('nc,nc->n', [query, positive_key]).unsqueeze(-1)
        
        # Compute negative logits: (batch_size, queue_size)
        l_neg = torch.einsum('nc,ck->nk', [query, negative_keys.T])
        
        # Concatenate positive and negative logits: (batch_size, 1 + queue_size)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        # Labels: positive is always the first (index 0)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def forward_with_logits(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        positive_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Alternative interface where all keys are provided with a mask.
        
        Args:
            query: Query embeddings of shape (batch_size, dim)
            keys: All key embeddings of shape (num_keys, dim)
            positive_mask: Boolean mask of shape (batch_size, num_keys)
                          indicating which keys are positive for each query
            
        Returns:
            Scalar loss value
        """
        # Normalize embeddings
        query = F.normalize(query, dim=1)
        keys = F.normalize(keys, dim=1)
        
        # Compute similarity logits: (batch_size, num_keys)
        logits = torch.einsum('nc,kc->nk', [query, keys])
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        # Find positive indices for each query
        # Assuming one positive per query
        labels = positive_mask.long().argmax(dim=1)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class MoCoLoss(nn.Module):
    """
    MoCo-specific loss wrapper.
    
    Handles the specific format of MoCo outputs where:
    - Query comes from the query encoder
    - Positive key comes from the momentum encoder (same sample, different augmentation)
    - Negative keys come from the queue
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter
        """
        super().__init__()
        self.info_nce = InfoNCELoss(temperature=temperature)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        queue: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MoCo loss.
        
        Args:
            query: Query embeddings from query encoder (batch_size, dim)
            key: Key embeddings from momentum encoder (batch_size, dim)
                 These are positive pairs for the queries
            queue: Negative key embeddings from queue (queue_size, dim)
            
        Returns:
            Scalar loss value
        """
        return self.info_nce(query, key, queue)


def compute_accuracy(
    query: torch.Tensor,
    positive_key: torch.Tensor,
    negative_keys: torch.Tensor,
    temperature: float = 0.07
) -> float:
    """
    Compute top-1 accuracy for contrastive learning.
    
    Measures how often the positive key has the highest similarity
    compared to negative keys.
    
    Args:
        query: Query embeddings of shape (batch_size, dim)
        positive_key: Positive key embeddings of shape (batch_size, dim)
        negative_keys: Negative key embeddings of shape (queue_size, dim)
        temperature: Temperature parameter
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    # Normalize embeddings
    query = F.normalize(query, dim=1)
    positive_key = F.normalize(positive_key, dim=1)
    negative_keys = F.normalize(negative_keys, dim=1)
    
    # Compute positive logits
    l_pos = torch.einsum('nc,nc->n', [query, positive_key]).unsqueeze(-1)
    
    # Compute negative logits
    l_neg = torch.einsum('nc,ck->nk', [query, negative_keys.T])
    
    # Concatenate logits
    logits = torch.cat([l_pos, l_neg], dim=1)
    logits = logits / temperature
    
    # Compute predictions (argmax)
    predictions = logits.argmax(dim=1)
    
    # Correct if prediction is 0 (positive key is at index 0)
    correct = (predictions == 0).float().sum()
    accuracy = correct / query.size(0)
    
    return accuracy.item()
