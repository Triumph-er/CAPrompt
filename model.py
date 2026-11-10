from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.15)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + residual


class ConfidencePromptCalibration(nn.Module):
    def __init__(
        self,
        context_dim: int,
        prompt_dim: int,
        init_scale: float = 0.05,
        conf_init: float = 0.05,
        dropout: float = 0.05,
        use_confidence: bool = True,
        use_affine: bool = True,
    ):
        super().__init__()
        hidden = max(context_dim, prompt_dim, 32)
        init_scale = float(min(max(init_scale, 1e-4), 1.0 - 1e-4))
        conf_init = float(min(max(conf_init, 1e-4), 1.0 - 1e-4))
        self.context_norm = nn.LayerNorm(context_dim * 2)
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, prompt_dim),
            nn.LayerNorm(prompt_dim),
        )
        self.prompt_norm = nn.LayerNorm(prompt_dim)
        self.gamma = nn.Linear(prompt_dim, prompt_dim)
        self.beta = nn.Linear(prompt_dim, prompt_dim)
        self.confidence = nn.Linear(prompt_dim, 1)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)
        nn.init.zeros_(self.confidence.weight)
        nn.init.constant_(self.confidence.bias, math.log(conf_init / (1.0 - conf_init)))
        self.scale_logit = nn.Parameter(torch.tensor(math.log(init_scale / (1.0 - init_scale))))
        self.dropout = nn.Dropout(dropout)
        self.use_confidence = bool(use_confidence)
        self.use_affine = bool(use_affine)

    def forward(
        self,
        context_tokens: torch.Tensor,
        prompt: torch.Tensor,
        return_stats: bool = False,
    ):
        # context_tokens: [B, C, N, 1], prompt: [B, P, N, 1]
        mean = context_tokens.mean(dim=2).squeeze(-1)
        std = context_tokens.var(dim=2, unbiased=False).add(1e-6).sqrt().squeeze(-1)
        context = torch.cat([mean, std], dim=-1)
        context = self.context_proj(self.context_norm(context)).unsqueeze(-1).unsqueeze(-1)

        prompt_n = prompt.squeeze(-1).transpose(1, 2)
        prompt_n = self.prompt_norm(prompt_n).transpose(1, 2).unsqueeze(-1)
        context_flat = context.squeeze(-1).squeeze(-1)
        gamma = torch.tanh(self.gamma(context_flat)).unsqueeze(-1).unsqueeze(-1)
        beta = torch.tanh(self.beta(context_flat)).unsqueeze(-1).unsqueeze(-1)
        if self.use_confidence:
            alpha = torch.sigmoid(self.confidence(context_flat)).view(-1, 1, 1, 1)
        else:
            alpha = torch.ones(
                context_flat.shape[0],
                1,
                1,
                1,
                dtype=context_flat.dtype,
                device=context_flat.device,
            )
        scale = torch.sigmoid(self.scale_logit)
        modulation = gamma * prompt_n + beta if self.use_affine else prompt_n
        residual = alpha * scale * self.dropout(modulation)
        if not return_stats:
            return residual
        return residual, {
            "alpha": alpha.flatten(1).mean(dim=1),
            "scale": scale.detach().expand(context_flat.shape[0]),
            "spatial_mean": mean.mean(dim=1),
            "spatial_std": std.mean(dim=1),
            "residual_norm": residual.flatten(1).norm(dim=1),
        }


class CrossDatasetSTID(nn.Module):
    def __init__(self, num_nodes_dict: Dict[str, int], **kwargs):
        super().__init__()
        self.num_nodes_dict = dict(num_nodes_dict)
        self.input_len = int(kwargs.get("input_len", 12))
        self.output_len = int(kwargs.get("output_len", 12))
        self.input_dim = int(kwargs.get("input_dim", 1))
        self.embed_dim = int(kwargs.get("embed_dim", 32))
        self.node_dim = int(kwargs.get("node_dim", 32))
        self.temp_dim_tid = int(kwargs.get("temp_dim_tid", 32))
        self.temp_dim_diw = int(kwargs.get("temp_dim_diw", 32))
        self.prompt_dim = int(kwargs.get("prompt_dim", 0))
        self.num_layers = int(kwargs.get("num_layers", 3))
        self.tod_size = int(kwargs.get("tod_size", 288))
        self.dow_size = int(kwargs.get("dow_size", 7))
        self.prompt_mode = kwargs.get("prompt_mode", "baseline")

        self.node_emb = nn.ParameterDict()
        for name, num_nodes in self.num_nodes_dict.items():
            p = nn.Parameter(torch.empty(num_nodes, self.node_dim))
            nn.init.xavier_uniform_(p)
            self.node_emb[name] = p

        self.time_in_day_emb = nn.Parameter(torch.empty(self.tod_size, self.temp_dim_tid))
        self.day_in_week_emb = nn.Parameter(torch.empty(self.dow_size, self.temp_dim_diw))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        if self.prompt_dim > 0 and self.prompt_mode in {"prompt", "confprompt"}:
            self.shared_prompt = nn.Parameter(torch.zeros(1, self.prompt_dim, 1, 1))
            nn.init.xavier_uniform_(self.shared_prompt)
        else:
            self.shared_prompt = None

        if self.prompt_mode == "confprompt":
            self.prompt_calibrator = ConfidencePromptCalibration(
                context_dim=self.embed_dim,
                prompt_dim=self.prompt_dim,
                init_scale=float(kwargs.get("prompt_init_scale", 0.05)),
                conf_init=float(kwargs.get("prompt_conf_init", 0.05)),
                dropout=float(kwargs.get("prompt_dropout", 0.05)),
                use_confidence=bool(kwargs.get("prompt_use_confidence", True)),
                use_affine=bool(kwargs.get("prompt_use_affine", True)),
            )
        else:
            self.prompt_calibrator = None

        hidden_dim = self.embed_dim + self.node_dim + self.temp_dim_tid + self.temp_dim_diw
        if self.shared_prompt is not None:
            hidden_dim += self.prompt_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(*[MultiLayerPerceptron(hidden_dim, hidden_dim) for _ in range(self.num_layers)])
        self.regression_layer = nn.Conv2d(hidden_dim, self.output_len, kernel_size=(1, 1), bias=True)

    @staticmethod
    def _feature_to_index(v: torch.Tensor, size: int) -> torch.Tensor:
        if v.detach().max().item() <= 1.0:
            idx = (v * size).long()
        else:
            idx = v.long()
        return idx.clamp(0, size - 1)

    def forward(
        self,
        history_data: torch.Tensor,
        dataset_name: str,
        return_aux: bool = False,
    ):
        # history_data: [B, T, N, C], output: [B, T_out, N, 1]
        input_data = history_data[..., : self.input_dim]
        batch_size, _, num_nodes, _ = input_data.shape

        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node = self.node_emb[dataset_name]
        if node.shape[0] != num_nodes:
            raise ValueError(f"{dataset_name} node count mismatch: model={node.shape[0]} batch={num_nodes}")
        node_emb = node.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)

        # tod = self._feature_to_index(history_data[..., 1][:, -1, :], self.tod_size)
        # dow = self._feature_to_index(history_data[..., 2][:, -1, :], self.dow_size)
        tod = self._feature_to_index(history_data[..., 1][:, -1, :],self.tod_size,)
        dow = (history_data[..., 2][:, -1, :].long().clamp(0, self.dow_size - 1))
        tod_emb = self.time_in_day_emb[tod].transpose(1, 2).unsqueeze(-1)
        dow_emb = self.day_in_week_emb[dow].transpose(1, 2).unsqueeze(-1)

        pieces = [time_series_emb, node_emb, tod_emb, dow_emb]
        aux = {}
        if self.shared_prompt is not None:
            prompt = self.shared_prompt.expand(batch_size, -1, num_nodes, -1)
            if self.prompt_calibrator is not None:
                if return_aux:
                    prompt_residual, aux = self.prompt_calibrator(
                        time_series_emb,
                        prompt,
                        return_stats=True,
                    )
                else:
                    prompt_residual = self.prompt_calibrator(time_series_emb, prompt)
                prompt = prompt + prompt_residual
            pieces.append(prompt)

        hidden = torch.cat(pieces, dim=1)
        hidden = self.encoder(hidden)
        pred = self.regression_layer(hidden)
        if return_aux:
            return pred, aux
        return pred
