import torch
from torch import nn
import torch.nn.functional as F

from .rmr import RMREncoder


class MAENet(nn.Module):
    """ Masked Auto-Encoder SSL model"""

    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 encoder_dropout: float = 0.25,
                 head_dropout: float = 0.5):

        super().__init__()
        self.encoder = RMREncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                  dropout=encoder_dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 2*hidden_dim),
                                  nn.SiLU(),
                                  nn.Dropout(head_dropout),
                                  nn.Linear(2*hidden_dim, input_dim))
    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)


class OSCNet(nn.Module):
    """ Online Syllable Clustering SSL model"""

    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 encoder_dropout: float = 0.25,
                 num_clusters=1024):

        super().__init__()
        self.encoder = RMREncoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers,
                                  dropout=encoder_dropout)

        self.proj = nn.Sequential(nn.Linear(hidden_dim, 2048),
                                 nn.SiLU(),
                                 nn.Linear(2048, 2048),
                                 nn.SiLU(),
                                 nn.Linear(2048, 256))

        v = torch.randn(num_clusters, 256)
        v = v / v.norm(2, dim=1, keepdim=True)
        self.v = nn.Parameter(v)

    def forward(self, x):
        x = self.encoder(x)
        x = self.proj(x)
        x = x.flatten(0, 1)  # B, T, D -> B*T, D
        x = nn.functional.normalize(x, dim=1, p=2, eps=1e-8)
        v = nn.functional.normalize(self.v, dim=1, p=2, eps=1e-8)
        return x @ v.T


class ClusterLoss(nn.Module):
    def __init__(self, student_temp=0.1, teacher_temp=0.04, reg_w=1.0):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.reg_w = reg_w

    @torch.no_grad()
    def sinkhorn_knopp(self, target_probs, iterations=3):
        Q = target_probs.float().t()  # B=samples, C=classes -> C, B
        Q /= torch.sum(Q)
        C, B = Q.shape
        for _ in range(iterations):
            # sum of samples' weights for each prototypes must be 1 / C
            Q /= (torch.sum(Q, dim=1, keepdim=True) * C)
            # sum of prototypes' weights for each sample must be 1 / B
            Q /= (torch.sum(Q, dim=0, keepdim=True) * B)
        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def gini_impurity(self, p):
        """
        p - student probabilities averaged on batch-axis (C,)
        """
        return 1.0 - (p ** 2).sum()

    def forward(self, student_logits, teacher_logits, student_temp=None, teacher_temp=None):
        student_temp = self.student_temp if student_temp is None else student_temp
        teacher_temp = self.teacher_temp if teacher_temp is None else teacher_temp
        student_lsm = F.log_softmax(student_logits / student_temp, dim=1)
        teacher_probs = F.softmax(teacher_logits / teacher_temp, dim=1)
        teacher_probs = self.sinkhorn_knopp(teacher_probs)
        # cross-entropy
        xe_loss = -(teacher_probs * student_lsm).sum(dim=1).mean()
        # we want to maximize gini-impurity
        gini_loss = -self.gini_impurity(student_lsm.exp().mean(dim=0))
        loss = xe_loss + self.reg_w * gini_loss
        return loss, xe_loss, gini_loss