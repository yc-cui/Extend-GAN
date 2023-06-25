import torch
from torch import nn
from torch.nn import functional as F


class LARN(nn.Module):
    def __init__(self, n_features, epsilon=1e-5):
        super(LARN, self).__init__()
        self.epsilon = epsilon
        self.gamma = torch.nn.Parameter(torch.randn((1, n_features, 1, 1)), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.randn((1, n_features, 1, 1)), requires_grad=True)


    def forward(self, x, m):
        _t = self._compute_T(x, m)
        context_feat = _t * m
        preserved_feat = x * (1. - m)
        return context_feat + preserved_feat

    def _compute_T(self, x, m):
        # q for known, 1 denotes invalid
        X_p = x * m
        X_q = x * (1. - m)
        X_p_mean = self._compute_weighted_mean(X_p, m)
        X_p_std = self._compute_weighted_std(X_p, m, X_p_mean)
        X_q_mean = self._compute_weighted_mean(X_q, (1. - m))
        X_q_std = self._compute_weighted_std(X_q, (1. - m), X_q_mean)
        return ((X_p - X_p_mean) / X_p_std) * X_q_std * self.gamma + X_q_mean * self.beta

    def _compute_weighted_mean(self, x, m):
        return torch.sum(x * m, dim=(2, 3), keepdim=True) / (torch.sum(m, dim=(2, 3), keepdim=True) + self.epsilon)

    def _compute_weighted_std(self, x, m, _mean):
        return torch.sqrt((torch.sum(torch.pow(x * m - _mean, 2), dim=(2, 3), keepdim=True) /
                          (torch.sum(m, dim=(2, 3), keepdim=True) + self.epsilon)) + self.epsilon)


if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    m = (torch.randn((8, 1, 256, 256)) > 0).float()
    l = LARN(3)
    o = l(x, m)
    print(o.shape)