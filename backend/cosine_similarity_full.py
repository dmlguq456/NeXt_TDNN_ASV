
import torch
import torch.nn.functional as F
import numpy as np
from typing import List


def cosine_similarity_full(enroll_embedding, test_embedding):
    """
    calculate cosine similarity
    Ref1 code: https://github.com/seongmin-kye/meta-SR/blob/master/EER_full.py
    Ref2 code: https://github.com/zyzisyz/mfa_conformer/blob/master/main.py
    Args:
        enroll_embedding (torch tensor): (batch, embedding_size)
        test_embedding (torch tensor): (batch, embedding_size)
    Return:
        score (float): cosine similarity for scalar
    """
    assert enroll_embedding.shape == test_embedding.shape, "enroll_embedding and test_embedding should be same shape"
    assert enroll_embedding.shape[0] == 1, "batch size should be 1"
    assert test_embedding.shape[0] == 1, "batch size should be 1"

    # normalize
    enroll_embedding_norm = F.normalize(enroll_embedding, p=2, dim=1) # (batch, embedding_size)
    test_embedding_norm = F.normalize(test_embedding, p=2, dim=1) # (batch, embedding_size)

    # dot product
    dot_product = torch.mm(enroll_embedding_norm, test_embedding_norm.transpose(0, 1)) # (batch, embedding_size) * (embedding_size, batch) -> batch * batch

    score = torch.mean(dot_product)

    return score