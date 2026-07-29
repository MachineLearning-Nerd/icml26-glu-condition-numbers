"""Direct CPU reproduction of the Section 5 CIFAR-10/Mixer experiment."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import energy_distance, t
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEEDS = (51, 52, 53)
EPOCHS = 15
BATCH_SIZE = 256


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FeedForward(nn.Module):
    """Official repository MLP, including its parameter-matching rule."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, activation: str):
        super().__init__()
        self.activation = activation
        if activation == "relu":
            self.fc1 = nn.Linear(in_dim, hidden_dim)
        elif activation == "reglu":
            hidden_dim = int(hidden_dim * (in_dim + out_dim) / (2 * in_dim + out_dim))
            self.fc1_a = nn.Linear(in_dim, hidden_dim)
            self.fc1_b = nn.Linear(in_dim, hidden_dim)
        else:
            raise ValueError(activation)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            hidden = F.relu(self.fc1(x))
        else:
            hidden = self.fc1_a(x) * F.relu(self.fc1_b(x))
        return self.fc2(hidden)


class MixerLayer(nn.Module):
    def __init__(self, dim: int, patches: int, activation: str):
        super().__init__()
        self.token_norm = nn.LayerNorm(dim)
        self.token_ff = FeedForward(patches, 4 * dim, patches, activation)
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_ff = FeedForward(dim, int(0.5 * dim), dim, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token = self.token_norm(x).transpose(1, 2)
        token = self.token_ff(token).transpose(1, 2)
        x = x + token
        return x + self.channel_ff(self.channel_norm(x))


class MLPMixer(nn.Module):
    """Paper code's CIFAR-10 model: patch=4, dim=256, depth=4."""

    def __init__(self, activation: str):
        super().__init__()
        self.patch = 4
        self.patch_embed = nn.Linear(3 * self.patch * self.patch, 256)
        self.layers = nn.ModuleList([MixerLayer(256, 64, activation) for _ in range(4)])
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(x.shape[0], 64, 3 * p * p)
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x).mean(dim=1))


def _copy_common_weights(source: nn.Module, target: nn.Module) -> int:
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied = 0
    for name, value in target_state.items():
        if name in source_state and source_state[name].shape == value.shape:
            target_state[name] = source_state[name].clone()
            copied += value.numel()
    target.load_state_dict(target_state)
    return copied


def _datasets() -> tuple:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1),
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    train = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    train_eval = datasets.CIFAR10("./data", train=True, download=False, transform=eval_transform)
    test = datasets.CIFAR10("./data", train=False, download=True, transform=eval_transform)
    return train, train_eval, test


def _loader(dataset, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.inference_mode()
def _loss(model: nn.Module, dataset) -> float:
    model.eval()
    total = 0.0
    count = 0
    for inputs, targets in _loader(dataset, shuffle=False, seed=0):
        logits = model(inputs)
        total += float(F.cross_entropy(logits, targets, reduction="sum"))
        count += targets.numel()
    return total / count


def _train_one(
    model: nn.Module,
    activation: str,
    seed: int,
    train,
    train_eval,
    test,
) -> list[dict]:
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=0.0)
    rows = []

    for epoch in range(EPOCHS + 1):
        train_loss = _loss(model, train_eval)
        test_loss = _loss(model, test)
        row = {
            "seed": seed,
            "activation": activation,
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "generalization_gap": test_loss - train_loss,
        }
        rows.append(row)
        print(
            "GAP_PROGRESS "
            f"seed={seed} activation={activation} epoch={epoch} "
            f"train={train_loss:.7f} test={test_loss:.7f} gap={test_loss-train_loss:.7f}",
            flush=True,
        )
        if epoch == EPOCHS:
            break

        # Re-create the stochastic stream so paired activations see identical
        # crop/flip/erase decisions and minibatch order at a given seed/epoch.
        epoch_seed = seed * 10_000 + epoch
        _seed(epoch_seed)
        model.train()
        for inputs, targets in _loader(train, shuffle=True, seed=epoch_seed):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
    return rows


def _conditional_differences(rows: list[dict]) -> tuple[list[dict], list[float]]:
    points = []
    seed_means = []
    for seed in SEEDS:
        curves = {}
        for activation in ("relu", "reglu"):
            selected = [r for r in rows if r["seed"] == seed and r["activation"] == activation]
            # Interpolate gap as a function of loss after sorting and collapsing
            # any numerically repeated loss values.
            selected.sort(key=lambda r: r["train_loss"])
            xs = np.asarray([r["train_loss"] for r in selected])
            ys = np.asarray([r["generalization_gap"] for r in selected])
            unique_x, unique_indices = np.unique(xs, return_index=True)
            curves[activation] = (unique_x, ys[unique_indices])
        low = max(curves["relu"][0].min(), curves["reglu"][0].min())
        high = min(curves["relu"][0].max(), curves["reglu"][0].max())
        grid = np.linspace(low, high, 11)
        diffs = []
        for train_loss in grid:
            non_gap = float(np.interp(train_loss, *curves["relu"]))
            glu_gap = float(np.interp(train_loss, *curves["reglu"]))
            diff = glu_gap - non_gap
            points.append(
                {
                    "seed": seed,
                    "matched_train_loss": float(train_loss),
                    "non_glu_gap": non_gap,
                    "glu_gap": glu_gap,
                    "glu_minus_non_gap": diff,
                }
            )
            diffs.append(diff)
        seed_means.append(float(np.mean(diffs)))
    return points, seed_means


def _paper_style_energy_test(rows: list[dict], permutations: int = 5000) -> dict:
    groups = []
    for activation in ("relu", "reglu"):
        selected = [r for r in rows if r["activation"] == activation]
        groups.append(
            np.asarray([[r["train_loss"], r["generalization_gap"]] for r in selected])
        )
    observed = float(energy_distance(groups[0].ravel(), groups[1].ravel()))
    combined = np.vstack(groups)
    n = len(groups[0])
    rng = np.random.default_rng(260520749)
    hits = 0
    for _ in range(permutations):
        order = rng.permutation(len(combined))
        permuted = energy_distance(combined[order[:n]].ravel(), combined[order[n:]].ravel())
        hits += permuted >= observed
    return {
        "implementation": "official plot script's flattened-coordinate energy distance",
        "observed": observed,
        "permutations": permutations,
        "p_value": (hits + 1) / (permutations + 1),
        "limitation": "epochs within a seed are dependent; this is secondary evidence only",
    }


def summarize(rows: list[dict]) -> dict:
    matched, seed_diffs = _conditional_differences(rows)
    mean_diff = float(np.mean(seed_diffs))
    sem = float(np.std(seed_diffs, ddof=1) / math.sqrt(len(seed_diffs)))
    radius = float(t.ppf(0.975, len(seed_diffs) - 1) * sem)
    optimization = []
    for seed in SEEDS:
        auc = {}
        for activation in ("relu", "reglu"):
            curve = [
                r["train_loss"]
                for r in rows
                if r["seed"] == seed and r["activation"] == activation
            ]
            auc[activation] = float(np.trapezoid(curve))
        optimization.append(
            {
                "seed": seed,
                "relu_loss_auc": auc["relu"],
                "reglu_loss_auc": auc["reglu"],
                "reglu_over_relu_auc": auc["reglu"] / auc["relu"],
                "reglu_faster": auc["reglu"] < auc["relu"],
            }
        )
    return {
        "matched_loss_points": matched,
        "seed_mean_gap_differences": seed_diffs,
        "mean_glu_minus_non_gap": mean_diff,
        "t_95pct_ci": [mean_diff - radius, mean_diff + radius],
        "predeclared_practical_margin_cross_entropy": 0.10,
        "optimization": optimization,
        "paper_style_energy_test": _paper_style_energy_test(rows),
    }


def run_generalization_gap() -> dict:
    started = time.perf_counter()
    torch.set_num_threads(8)
    train, train_eval, test = _datasets()
    rows = []
    parameter_counts = {}
    copied_common_parameters = {}
    for seed in SEEDS:
        _seed(seed)
        relu = MLPMixer("relu")
        _seed(seed + 1000)
        reglu = MLPMixer("reglu")
        copied_common_parameters[str(seed)] = _copy_common_weights(relu, reglu)
        models = {"relu": relu, "reglu": reglu}
        for activation, model in models.items():
            parameter_counts[activation] = sum(p.numel() for p in model.parameters())
            rows.extend(_train_one(model, activation, seed, train, train_eval, test))
    return {
        "claim": "Section 5/Figure 7: optimization advantage with limited matched-loss gap advantage",
        "paper_anchor": "#S5 and #S5.F7",
        "official_code_anchor": "train_activation_gap.py@cc1664bf48505d7bac308b308ce1c495b06ce979",
        "protocol": {
            "dataset": "full CIFAR-10: 50,000 train and 10,000 test",
            "model": "official MLP-Mixer patch=4, dim=256, depth=4",
            "activations": ["relu", "reglu"],
            "optimizer": "SGD lr=0.005 momentum=0.9 weight_decay=0",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "seeds": list(SEEDS),
            "torch_threads": 8,
            "train_evaluation": "all 50,000 unaugmented training examples",
            "test_evaluation": "all 10,000 test examples",
        },
        "dataset_sizes": {"train": len(train_eval), "test": len(test)},
        "parameter_counts": parameter_counts,
        "copied_common_parameters": copied_common_parameters,
        "rows": rows,
        "summary": summarize(rows),
        "negative_control": {
            "construction": "subtract 0.50 CE from every matched ReGLU gap",
            "intended_rejection": "large artificial GLU generalization advantage",
            "shift": -0.50,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
