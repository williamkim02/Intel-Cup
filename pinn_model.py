"""
pinn_model.py  -  SOH PINN 모델 클래스
"""
import torch
import torch.nn as nn


class SOHCurvePINN(nn.Module):
    """
    4개 feature -> SOH 예측.
    sigmoid 출력으로 SOH in (0, 1) 자동 보장.
    """
    def __init__(self, n_features=4, hidden_dim=32, hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(n_features, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))
