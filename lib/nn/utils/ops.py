import torch
import torch.nn.functional as F
from einops import reduce
from torch.autograd import Variable

from ... import epsilon


def mae(y_hat, y, reduction='none'):
    return F.l1_loss(y_hat, y, reduction=reduction)


def mape(y_hat, y):
    return torch.abs((y_hat - y) / y)


def wape_loss(y_hat, y):
    l = torch.abs(y_hat - y)
    return l.sum() / (y.sum() + epsilon)


def smape_loss(y_hat, y):
    c = torch.abs(y) > epsilon
    l_minus = torch.abs(y_hat - y)
    l_plus = torch.abs(y_hat + y) + epsilon
    l = 2 * l_minus / l_plus * c.float()
    return l.sum() / c.sum()


def peak_prediction_loss(y_hat, y, reduction='none'):
    y_max = reduce(y, 'b s n 1 -> b 1 n 1', 'max')
    y_min = reduce(y, 'b s n 1 -> b 1 n 1', 'min')
    target = torch.cat([y_max, y_min], dim=1)
    return F.mse_loss(y_hat, target, reduction=reduction)


def wrap_loss_fn(base_loss):
    def loss_fn(y_hat, y_true, mask=None):
        scaling = 1.
        if mask is not None:
            try:
                loss = base_loss(y_hat, y_true, reduction='none')
            except TypeError:
                loss = base_loss(y_hat, y_true)
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + epsilon)
            # scaling = mask.sum() / torch.numel(mask)
        else:
            loss = base_loss(y_hat, y_true).mean()
        return scaling * loss

    return loss_fn


def rbf_sim(x, gamma, device='cpu'):
    n = x.size()[0]
    a = torch.exp(-gamma * F.pdist(x, 2) ** 2)
    row_idx, col_idx = torch.triu_indices(n, n, 1)
    A = 0.5 * torch.eye(n, n).to(device)
    A[row_idx, col_idx] = a
    return A + A.T


def reverse_tensor(tensor=None, axis=-1):
    if tensor is None:
        return None
    if tensor.dim() <= 1:
        return tensor
    indices = range(tensor.size()[axis])[::-1]
    indices = Variable(torch.LongTensor(indices), requires_grad=False).to(tensor.device)
    return tensor.index_select(axis, indices)


def block_diag(*arrs):
    shapes = torch.tensor([a.shape for a in arrs])
    total_shape = torch.sum(shapes, dim=0).tolist()
    device = arrs[0].device

    # Initialize lists to store the new indices and values
    new_indices = []
    new_values = []

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        indices = arrs[i]._indices()
        values = arrs[i]._values()

        # Adjust the indices for the block position
        new_indices.append(indices + torch.tensor([[r], [c]], device=device))
        new_values.append(values)

        r += rr
        c += cc

    # Concatenate the indices and values
    new_indices = torch.cat(new_indices, dim=1)
    new_values = torch.cat(new_values)

    # Create the final block diagonal sparse matrix
    out = torch.sparse_coo_tensor(new_indices, new_values, total_shape, device=device)

    return out
