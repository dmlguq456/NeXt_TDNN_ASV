
import torch
import torch.nn.functional as F
from typing import List



def euclidean_distance_full(enroll_embedding: torch.Tensor, test_embedding: torch.Tensor)->torch.FloatTensor:
    """
    calculate cosine similarity [BATCH x BATCH]
    Ref paper: In defence of metric learning for speaker recognition[https://arxiv.org/pdf/2003.11982.pdf]
    Ref paper2: VoxCeleb2: Deep Speaker Recognition [https://arxiv.org/pdf/1806.05622.pdf]
    Ref code: https://github.com/clovaai/voxceleb_trainer/blob/343af8bc9b325a05bcf28772431b83f5b8817f5a/SpeakerNet.py#L136
    Args:
        enroll_embedding (torch tensor): (BATCH, embedding_size)
        test_embedding (torch tensor): (BATCH, embedding_size)
    Return:
        score (float): cosine similarity 
    """
    assert enroll_embedding.shape == test_embedding.shape, "enroll_embedding and test_embedding should be same shape"
    assert enroll_embedding.shape[0] == 1, "num_seg should be greater than 1"
    assert test_embedding.shape[0] == 1, "num_seg should be greater than 1"

    enroll_embedding = F.normalize(enroll_embedding, p=2, dim=1) # L2 norm  (BATCH, embedding_size)
    test_embedding = F.normalize(test_embedding, p=2, dim=1) # L2 norm  (BATCH, embedding_size)
    
    distance = torch.cdist(enroll_embedding, test_embedding) # (BATCH, BATCH)
    score = -1 * torch.mean(distance) # -1 * mean distance to 

    return score # scalar
