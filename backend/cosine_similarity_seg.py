import torch
import torch.nn.functional as F

def cosine_similarity_seg(enroll_embedding: torch.Tensor, test_embedding: torch.Tensor)-> torch.FloatTensor:
    """
    calculate cosine similarity for num_seg x num_seg
    Args:
        enroll_embedding (torch tensor): (num_seg, embedding_size)
        test_embedding (torch tensor): (num_seg, embedding_size) or (1, embedding_size)
    Return:
        score (torch tensor): cosine similarity 
    """

    # for debug
    assert enroll_embedding.shape == test_embedding.shape, "enroll_embedding and test_embedding should be same shape"
    assert enroll_embedding.shape[0] != 1, "num_seg should be greater than 1"
    assert test_embedding.shape[0] != 1, "num_seg should be greater than 1"
    
    # normalize (||enroll_embedding||, ||test_embedding||)
    enroll_embedding_norm = F.normalize(enroll_embedding, p=2, dim=1) # (num_seg, embedding_size)
    test_embedding_norm = F.normalize(test_embedding, p=2, dim=1) # (num_seg, embedding_size)

    # dot product (dot_product(enroll_embedding_norm, test_embedding_norm))
    dot_product = torch.mm(enroll_embedding_norm, test_embedding_norm.transpose(0, 1)) # (num_seg, embedding_size) * (embedding_size, num_seg) -> num_seg * num_seg

    score = torch.mean(dot_product)

    return score # scalar

    