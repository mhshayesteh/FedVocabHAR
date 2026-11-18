"""Clean-room PyTorch implementation of Native LoGo (Kim et al., CVPR 2023,
arXiv:2303.12317, Section 5 & Appendix C Algorithm 1) adapted for 4-channel
128-sample wearable HAR windows under a B=4 query budget.

No upstream repository code is copied. Intended for a separate complete-system
comparison table (FedAvg + local-only retraining), distinct from frozen-encoder
FedVocabHAR tables.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class TemporalInceptionBlock(nn.Module):
    def __init__(self, in_ch: int, branch_ch: int = 32, kernels: Sequence[int] = (9, 19, 39)):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, branch_ch, kernel_size=1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(branch_ch, branch_ch, kernel_size=k, padding=k // 2, bias=False)
            for k in kernels
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, branch_ch, kernel_size=1, bias=False),
        )
        out_ch = branch_ch * (len(kernels) + 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.residual = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        branches = [conv(b) for conv in self.convs] + [self.pool_branch(x)]
        out = self.bn(torch.cat(branches, dim=1))
        return F.relu(out + self.residual(x))


class TemporalInceptionClassifier(nn.Module):
    """Matches FedVocabHAR's 4-channel Temporal Inception encoder + linear head."""

    def __init__(self, in_ch: int = 4, num_classes: int = 6, emb_dim: int = 64):
        super().__init__()
        self.block1 = TemporalInceptionBlock(in_ch, branch_ch=32, kernels=(9, 19, 39))
        self.block2 = TemporalInceptionBlock(128, branch_ch=32, kernels=(9, 19, 39))
        self.proj = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )
        self.head = nn.Linear(emb_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.block2(self.block1(x))
        h = h.mean(dim=-1)
        z = self.proj(h)
        return F.normalize(z, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.head(z)
        return logits, z


@dataclass
class ClientState:
    client_id: int
    x: np.ndarray          # (N_k, C, T)
    y_true: np.ndarray     # (N_k,) integer class IDs (used only by oracle/eval)
    window_ids: np.ndarray # (N_k,)
    labeled_mask: np.ndarray  # (N_k,) bool


def _train_model_on_client(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> None:
    if len(x) == 0:
        return
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits, _ = model(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt.step()


def fedavg_round(
    global_model: nn.Module,
    clients: Sequence[ClientState],
    local_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    base_state = copy.deepcopy(global_model.state_dict())
    weighted_states: List[Tuple[int, Dict[str, torch.Tensor]]] = []
    total_samples = 0

    for c in clients:
        idx = np.flatnonzero(c.labeled_mask)
        if len(idx) == 0:
            continue
        local_model = copy.deepcopy(global_model).to(device)
        _train_model_on_client(
            local_model, c.x[idx], c.y_true[idx],
            epochs=local_epochs, batch_size=batch_size, lr=lr, device=device,
        )
        weighted_states.append((len(idx), {k: v.detach().cpu() for k, v in local_model.state_dict().items()}))
        total_samples += len(idx)

    if total_samples == 0:
        return global_model

    avg_state = {}
    for k in base_state:
        acc = sum((n / total_samples) * st[k].float() for n, st in weighted_states)
        avg_state[k] = acc.to(base_state[k].dtype)
    global_model.load_state_dict(avg_state)
    return global_model


def _kmeans_np(feats: np.ndarray, k: int, seed: int = 0, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(feats)
    k = min(k, n)
    init_idx = rng.choice(n, size=k, replace=False)
    centers = feats[init_idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dists = np.sum((feats[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = feats[labels == j]
            if len(members) > 0:
                centers[j] = members.mean(axis=0)
    return labels, centers


@torch.no_grad()
def _infer_probs_and_emb(
    model: nn.Module, x: np.ndarray, batch_size: int = 256, device: torch.device = torch.device("cpu")
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all, emb_all = [], []
    for i in range(0, len(x), batch_size):
        bx = torch.from_numpy(x[i:i + batch_size]).float().to(device)
        logits, z = model(bx)
        probs_all.append(F.softmax(logits, dim=-1).cpu().numpy())
        emb_all.append(z.cpu().numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(emb_all, axis=0)


def logo_select_queries(
    global_model: nn.Module,
    clients: Sequence[ClientState],
    model_factory,
    budget: int = 4,
    local_epochs: int = 5,
    lr: float = 1e-3,
    seed: int = 0,
    device: torch.device = torch.device("cpu"),
) -> List[Tuple[int, int, float]]:
    """LoGo two-step selection (Macro: local-only gradient-embedding k-means;
    Micro: 1 EM step + global-model entropy per cluster), ranked globally up to
    `budget` queries. Returns [(client_idx, local_pos, entropy_score), ...].
    """
    candidates: List[Tuple[float, int, int, int]] = []  # (-entropy, client_id, window_id, client_idx, local_pos)
    ranked: List[Tuple[float, int, int, int, int]] = []

    for c_idx, c in enumerate(clients):
        unlabeled_idx = np.flatnonzero(~c.labeled_mask)
        labeled_idx = np.flatnonzero(c.labeled_mask)
        if len(unlabeled_idx) == 0 or len(labeled_idx) == 0:
            continue

        # Train local-only model from scratch on client's accumulated labels
        local_model = model_factory().to(device)
        _train_model_on_client(
            local_model, c.x[labeled_idx], c.y_true[labeled_idx],
            epochs=local_epochs, lr=lr, device=device,
        )

        # Macro step: predicted-class gradient embedding z * (1 - p_max)
        p_loc, z_loc = _infer_probs_and_emb(local_model, c.x[unlabeled_idx], device=device)
        p_max = p_loc.max(axis=1, keepdims=True)
        grad_emb = z_loc * (1.0 - p_max)

        k_clusters = min(budget, len(unlabeled_idx))
        macro_labels, macro_centers = _kmeans_np(grad_emb, k_clusters, seed=seed + c.client_id)

        # Micro step: 1 EM reassignment + global-model entropy per cluster
        p_glob, _ = _infer_probs_and_emb(global_model, c.x[unlabeled_idx], device=device)
        entropy = -np.sum(p_glob * np.log(np.clip(p_glob, 1e-12, 1.0)), axis=1)

        # E-step assignment to macro_centers
        dists = np.sum((grad_emb[:, None, :] - macro_centers[None, :, :]) ** 2, axis=2)
        assign = np.argmin(dists, axis=1)

        for cl in range(k_clusters):
            members = np.flatnonzero(assign == cl)
            if len(members) == 0:
                members = np.flatnonzero(macro_labels == cl)
            if len(members) == 0:
                continue
            best_m = max(members, key=lambda m: (float(entropy[m]), -int(c.window_ids[unlabeled_idx[m]])))
            pos = int(unlabeled_idx[best_m])
            ranked.append((-float(entropy[best_m]), c.client_id, int(c.window_ids[pos]), c_idx, pos))

    ranked.sort()
    selected = []
    for neg_ent, _, _, c_idx, pos in ranked[:budget]:
        selected.append((c_idx, pos, -neg_ent))
    return selected


def compute_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    known_classes: Sequence[int],
    novel_classes: Sequence[int],
) -> Dict[str, float]:
    def _f1_for(classes: Sequence[int]) -> float:
        scores = []
        for c in classes:
            tp = int(np.sum((y_true == c) & (y_pred == c)))
            fp = int(np.sum((y_true != c) & (y_pred == c)))
            fn = int(np.sum((y_true == c) & (y_pred != c)))
            denom = 2 * tp + fp + fn
            scores.append((2.0 * tp / denom) if denom > 0 else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    return {
        "known_macro_f1": _f1_for(known_classes),
        "novel_macro_f1": _f1_for(novel_classes),
        "combined_macro_f1": _f1_for(list(known_classes) + list(novel_classes)),
    }


def run_self_test() -> Dict[str, object]:
    """Synthetic smoke test with no external data."""
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    in_ch, seq_len, num_classes = 4, 128, 6
    known_classes = [0, 1, 2, 3]
    novel_classes = [4, 5]

    clients = []
    wid = 0
    for cid in range(3):
        xs, ys, wids, masks = [], [], [], []
        for cls in range(num_classes):
            for j in range(6):
                sig = rng.normal(cls * 0.5, 0.2, size=(in_ch, seq_len)).astype(np.float32)
                xs.append(sig)
                ys.append(cls)
                wids.append(wid)
                masks.append(cls in known_classes and j < 3)
                wid += 1
        clients.append(ClientState(cid, np.stack(xs), np.array(ys), np.array(wids), np.array(masks)))

    factory = lambda: TemporalInceptionClassifier(in_ch=in_ch, num_classes=num_classes, emb_dim=64)
    global_model = factory()
    for _ in range(2):
        fedavg_round(global_model, clients, local_epochs=2)

    selected = logo_select_queries(global_model, clients, factory, budget=4, local_epochs=2, seed=42)
    assert len(selected) == 4, f"Expected 4 queries, got {len(selected)}"
    for c_idx, pos, _ in selected:
        assert not clients[c_idx].labeled_mask[pos]
        clients[c_idx].labeled_mask[pos] = True

    fedavg_round(global_model, clients, local_epochs=2)
    x_test = np.concatenate([c.x for c in clients], axis=0)
    y_test = np.concatenate([c.y_true for c in clients], axis=0)
    probs, _ = _infer_probs_and_emb(global_model, x_test)
    metrics = compute_macro_f1(y_test, probs.argmax(axis=1), known_classes, novel_classes)
    return {"passed": True, "selected_count": len(selected), "metrics": metrics}


if __name__ == "__main__":
    print(json.dumps(run_self_test(), indent=2))
