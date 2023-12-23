
import torch
import torch.nn.functional as F
from typing import List


def euclidean_distance_mix(enroll_embedding_full: torch.Tensor, test_embedding_full: torch.Tensor, embedding_seg:torch.Tensor, test_embedding_seg: torch.Tensor)->torch.FloatTensor:
    """
    calculate euclidean distance
    Args:
        enroll_embedding_full (torch tensor): (batch, embedding_size)
        test_embedding_full (torch tensor): (batch, embedding_size)
        embedding_seg (torch tensor): (num_seg, embedding_size)        
        test_embedding_seg (torch tensor): (num_seg, embedding_size)
    Return:
        score (torch tensor): euclidean distance for scalar    
    """

    assert enroll_embedding_full.shape == test_embedding_full.shape, "enroll_embedding_full and test_embedding_full should be same shape"
    assert enroll_embedding_full.shape[0] == 1, "batch size should be 1"

    # normalize 
    enroll_embedding_full_norm = F.normalize(enroll_embedding_full, p=2, dim=1) # L2 norm (batch, embedding_size)
    test_embedding_full_norm = F.normalize(test_embedding_full, p=2, dim=1) # L2 norm (batch, embedding_size)
    embedding_seg_norm = F.normalize(embedding_seg, p=2, dim=1) # L2 norm (num_seg, embedding_size)
    test_embedding_seg_norm = F.normalize(test_embedding_seg, p=2, dim=1) # L2 norm (num_seg, embedding_size)

    distance_full = torch.cdist(enroll_embedding_full_norm, test_embedding_full_norm) # (batch, batch)
    distance_seg = torch.cdist(embedding_seg_norm, test_embedding_seg_norm) # (num_seg, num_seg)

    score = torch.mean(distance_full) + torch.mean(distance_seg)
    score = (-1 * score) / 2

    return score # scalar