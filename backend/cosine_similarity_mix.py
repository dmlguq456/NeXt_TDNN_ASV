import torch
import torch.nn.functional as F
from typing import List


def cosine_similarity_mix(enroll_embedding_full: torch.Tensor, test_embedding_full: torch.Tensor, embedding_seg:torch.Tensor, test_embedding_seg: torch.Tensor)->torch.FloatTensor:
    """
    calculate cosine similarity for scalar
    Ref: https://github.com/TaoRuijie/ECAPA-TDNN/blob/bd12fb2e5a8c37c0329836002d6ef7f026d5d9d0/ECAPAModel.py#L46
    Args:
        enroll_embedding_full (torch tensor): (batch, embedding_size)
        embedding_seg (torch tensor): (num_seg, embedding_size)
        test_embedding_full (torch tensor): (batch, embedding_size)
        test_embedding_seg (torch tensor): (num_seg, embedding_size)
    Return:
        score (torch tensor): cosine similarity
    """
    # normalize
    enroll_embedding_full_norm = F.normalize(enroll_embedding_full, p=2, dim=1) # (BATCH x embedding_size)
    test_embedding_full_norm = F.normalize(test_embedding_full, p=2, dim=1) # (BATCH x embedding_size)
    embedding_seg_norm = F.normalize(embedding_seg, p=2, dim=1) # (numseg, embedding_size)
    test_embedding_seg_norm = F.normalize(test_embedding_seg, p=2, dim=1) # (numseg, embedding_size)

    # dot product
    dot_product_full = torch.mm(enroll_embedding_full_norm, test_embedding_full_norm.transpose(0, 1)) # (BATCH x embedding_size) * (embedding_size x BATCH) -> BATCH x BATCH
    dot_product_seg = torch.mm(embedding_seg_norm, test_embedding_seg_norm.transpose(0, 1)) # (numseg x embedding_size) * (embedding_size x numseg) -> numseg x numseg

    score = torch.mean(dot_product_full) + torch.mean(dot_product_seg)
    score = score / 2

    return score
