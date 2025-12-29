import torch


def contrastive_loss(args, hidden, nbr_idx, neg_idx, train_idx):
    if not args.do_contrast:
        return 0
    loss_con = 0
    for i in range(len(train_idx)):
        index = train_idx[i]
        if int(index) < len(train_idx) - 1:
            hidden_positive = hidden[nbr_idx[index]]
            positive = torch.exp(torch.cosine_similarity(hidden[i].unsqueeze(0), hidden_positive.detach()))
            hidden_negative = hidden[neg_idx[index]]
            negative = torch.exp(torch.cosine_similarity(hidden[i].unsqueeze(0), hidden_negative.detach())).sum()
            loss_con -= torch.log((positive / negative)).sum()
            torch.cuda.empty_cache()
    return loss_con / len(train_idx)


def orthogonal_loss(shared, specific):
    _shared = shared.detach()
    _shared = _shared - _shared.mean(dim=0)
    correlation_matrix = _shared.t().matmul(specific)
    norm = torch.norm(correlation_matrix, p=1)
    return norm