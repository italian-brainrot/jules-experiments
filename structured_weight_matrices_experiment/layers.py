import torch
import torch.nn as nn
import torch.nn.functional as F

class KroneckerLinear(nn.Module):
    def __init__(self, m1, n1, m2, n2, bias=True):
        super().__init__()
        self.m1, self.n1, self.m2, self.n2 = m1, n1, m2, n2
        # W = A \otimes B
        # A is (m1, n1), B is (m2, n2)
        # Initialize with Kaiming-like scaling: Var(W) = 2/in_features
        # Var(A)Var(B) = 2/(n1*n2)
        std = (2.0 / (n1 * n2))**0.25
        self.A = nn.Parameter(torch.randn(m1, n1) * std)
        self.B = nn.Parameter(torch.randn(m2, n2) * std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(m1 * m2))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: (batch, n1 * n2)
        batch_size = x.shape[0]
        # Reshape x to (batch, n1, n2)
        X = x.view(batch_size, self.n1, self.n2)
        # Y = A X B^T
        # A: (m1, n1), X: (batch, n1, n2), B^T: (n2, m2)
        Y = torch.matmul(self.A, X) # (batch, m1, n2)
        Y = torch.matmul(Y, self.B.t()) # (batch, m1, m2)
        y = Y.reshape(batch_size, -1)
        if self.bias is not None:
            y = y + self.bias
        return y

class LowRankDiagonalLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        # Initialize so that Var(W) approx 2/in_features
        std = (2.0 / in_features)**0.5
        # For rank-1, Var(UV^T) = Var(U)Var(V) = std^2
        self.U = nn.Parameter(torch.randn(out_features, rank) * (std**0.5))
        self.V = nn.Parameter(torch.randn(in_features, rank) * (std**0.5))
        self.diag = nn.Parameter(torch.randn(min(in_features, out_features)) * std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # U V^T x = U (V^T x)
        v_t_x = F.linear(x, self.V.t()) # (batch, rank)
        u_v_t_x = F.linear(v_t_x, self.U) # (batch, out_features)

        # Dx
        k = len(self.diag)
        dx_base = x[:, :k] * self.diag

        if self.out_features > self.in_features:
             res_dx = torch.zeros(x.shape[0], self.out_features, device=x.device, dtype=x.dtype)
             res_dx[:, :k] = dx_base
        else:
             res_dx = dx_base[:, :self.out_features]

        y = u_v_t_x + res_dx
        if self.bias is not None:
            y = y + self.bias
        return y

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, density=0.1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.density = density
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (2.0 / in_features)**0.5)
        mask = (torch.rand(out_features, in_features) < density).float()
        # Scale weight to compensate for density
        self.weight.data /= (density**0.5)
        self.register_buffer('mask', mask)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        w = self.weight * self.mask
        return F.linear(x, w, self.bias)
