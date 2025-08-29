import torch.nn.functional as F

def kl_divergence(p_logits, q_logits):
    """
    Compute KL divergence D_KL(P || Q) for logits.
    """
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    kl = F.kl_div(p, q, reduction='batchmean')
    return kl
