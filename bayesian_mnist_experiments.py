"""
bayesian_mnist_experiments.py
Jiadi Wu
Course final project: Bayesian neural networks with weight priors for MNIST.

Experiments:
  (1) Main comparison on full MNIST: MAP (L2) vs MC-Dropout vs Bayes-by-Backprop (BBB)
      Metrics: test accuracy, NLL, ECE (15 bins).
  (2) Robustness / calibration under input corruption (Gaussian noise) and pure-noise OOD.
  (3) Small-data adaptation: training on 5,000 examples.

Outputs:
  results/results.json      - all numeric results
  results/fig_*.png         - figures for the report
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(2)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 256
LR = 1e-3
EPOCHS_FULL = 12
EPOCHS_SMALL = 15
MC_SAMPLES = 50          # stochastic forward passes at test time
ECE_BINS = 15
SWEEP_EPOCHS = 8         # epochs for validation-based hyper-parameter selection


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_mnist():
    from torchvision import datasets, transforms

    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tr)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tr)

    Xtr = train.data.float().view(-1, 784) / 255.0
    ytr = train.targets
    Xte = test.data.float().view(-1, 784) / 255.0
    yte = test.targets

    # standardize with training statistics
    m = Xtr.mean(dim=0, keepdim=True)
    s = Xtr.std(dim=0, keepdim=True) + 1e-8
    Xtr = (Xtr - m) / s
    Xte = (Xte - m) / s

    # train / validation split (10% of train for hyper-parameter selection)
    n_val = 6000
    Xva, yva = Xtr[:n_val], ytr[:n_val]
    Xtrn, ytrn = Xtr[n_val:], ytr[n_val:]

    # small-data subset (first 5000 of the training portion)
    Xsm, ysm = Xtrn[:5000], ytrn[:5000]

    return (Xtrn, ytrn), (Xva, yva), (Xte, yte), (Xsm, ysm)


def make_loader(X, y, shuffle=True):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class MAPNet(nn.Module):
    """Standard MLP trained with cross-entropy + L2 (Gaussian-prior MAP)."""

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(784, 128), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class BayesLinear(nn.Module):
    """Mean-field Gaussian variational layer (Bayes by Backprop)."""

    def __init__(self, in_f, out_f, prior_std=1.0):
        super().__init__()
        self.prior_std = prior_std
        self.mu = nn.Parameter(torch.empty(in_f, out_f).uniform_(-0.2, 0.2))
        self.rho = nn.Parameter(torch.full((in_f, out_f), np.log(np.expm1(0.1))))

    def forward(self, x):
        sigma = F.softplus(self.rho)
        eps = torch.randn_like(self.mu)
        w = self.mu + sigma * eps
        # KL between q(w) = N(mu, sigma^2) and p(w) = N(0, prior_std^2)
        kl = 0.5 * (torch.log(self.prior_std ** 2 / sigma ** 2)
                    + (sigma ** 2 + self.mu ** 2) / self.prior_std ** 2 - 1)
        kl = kl.sum()
        return F.linear(x, w.t()), kl

    def expected_weights(self):
        return self.mu


class BBBNet(nn.Module):
    """Bayes-by-Backprop MLP (Blundell et al., 2015), prior N(0, 1)."""

    def __init__(self, prior_std=1.0):
        super().__init__()
        self.l1 = BayesLinear(784, 128, prior_std)
        self.l2 = BayesLinear(128, 128, prior_std)
        self.l3 = BayesLinear(128, 10, prior_std)

    def forward(self, x):
        kl = 0.0
        h, k = self.l1(x)
        kl = kl + k
        h = F.relu(h)
        h, k = self.l2(h)
        kl = kl + k
        h = F.relu(h)
        logits, k = self.l3(h)
        kl = kl + k
        return logits, kl


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_map(model, loader, epochs, l2=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=l2)
    hist = []
    for ep in range(epochs):
        model.train()
        tot, corr, n = 0.0, 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            corr += (out.argmax(1) == yb).sum().item()
            n += yb.size(0)
        hist.append(corr / n)
    return hist


def train_bbb(model, loader, epochs, n_train):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist = []
    for ep in range(epochs):
        model.train()
        tot, corr, n = 0.0, 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits, kl = model(xb)
            nll = F.cross_entropy(logits, yb, reduction="sum")
            loss = nll + kl * (xb.size(0) / n_train)   # per-sample ELBO scaling
            loss.backward()
            opt.step()
            corr += (logits.argmax(1) == yb).sum().item()
            n += yb.size(0)
        hist.append(corr / n)
    return hist


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def predict_map(model, X, batch=512):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch):
            probs.append(F.softmax(model(X[i:i + batch]), dim=1))
    return torch.cat(probs)


def predict_mc_dropout(model, X, T=MC_SAMPLES, batch=512):
    model.train()   # keep dropout active
    acc = []
    with torch.no_grad():
        for _ in range(T):
            ps = []
            for i in range(0, X.size(0), batch):
                ps.append(F.softmax(model(X[i:i + batch]), dim=1))
            acc.append(torch.cat(ps))
    return torch.stack(acc).mean(0)


def predict_bbb(model, X, T=MC_SAMPLES, batch=512):
    model.eval()
    acc = []
    with torch.no_grad():
        for _ in range(T):
            ps = []
            for i in range(0, X.size(0), batch):
                logits, _ = model(X[i:i + batch])
                ps.append(F.softmax(logits, dim=1))
            acc.append(torch.cat(ps))
    return torch.stack(acc).mean(0)


def metrics(probs, y):
    pred = probs.argmax(1)
    acc = (pred == y).float().mean().item()
    conf = probs.gather(1, y.view(-1, 1)).squeeze()
    nll = -torch.log(conf + 1e-12).mean().item()
    # ECE
    bin_bounds = torch.linspace(0, 1, ECE_BINS + 1)
    ece = 0.0
    for b in range(ECE_BINS):
        lo, hi = bin_bounds[b], bin_bounds[b + 1]
        m = (conf >= lo) & (conf < hi)
        if b == ECE_BINS - 1:
            m = (conf >= lo) & (conf <= hi)
        if m.sum() > 0:
            acc_b = (pred[m] == y[m]).float().mean().item()
            conf_b = conf[m].mean().item()
            ece += m.float().mean().item() * abs(acc_b - conf_b)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(1).mean().item()
    return {"acc": acc, "nll": nll, "ece": ece, "entropy": entropy}


def reliability_data(probs, y, bins=15):
    pred = probs.argmax(1)
    conf = probs.gather(1, y.view(-1, 1)).squeeze()
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf >= lo) & (conf < hi)
        if b == bins - 1:
            m = (conf >= lo) & (conf <= hi)
        if m.sum() > 0:
            out.append((lo, hi, (pred[m] == y[m]).float().mean().item(), conf[m].mean().item(), m.float().mean().item()))
    return out


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_train_curves(hist_map, hist_mcd, hist_bbb, path):
    plt.figure(figsize=(5.2, 3.4))
    xs = np.arange(1, len(hist_map) + 1)
    plt.plot(xs, np.array(hist_map) * 100, "o-", label="MAP (L2)", lw=1.5)
    plt.plot(xs, np.array(hist_mcd) * 100, "s--", label="MC-Dropout", lw=1.5)
    plt.plot(xs, np.array(hist_bbb) * 100, "^-", label="Bayes by Backprop", lw=1.5)
    plt.xlabel("Epoch"); plt.ylabel("Training accuracy (%)")
    plt.legend(frameon=False); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def fig_reliability(rel_map, rel_mcd, rel_bbb, path):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    for ax, (rel, name) in zip(axes, [(rel_map, "MAP (L2)"), (rel_mcd, "MC-Dropout"), (rel_bbb, "Bayes by Backprop")]):
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        confs = [r[3] for r in rel]; accs = [r[2] for r in rel]; ws = [r[4] for r in rel]
        ax.bar(confs, accs, width=[w * 1.0 for w in ws], alpha=0.7, color="#4C72B0", align="center")
        ax.set_title(name, fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontsize=8); ax.set_ylabel("Accuracy", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Reliability diagrams on the clean test set", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close()


def fig_entropy_corruption(noise_levels, ent_map, ent_mcd, ent_bbb, ood_ent, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    ax = axes[0]
    ax.plot(noise_levels, ent_map, "o-", label="MAP (L2)")
    ax.plot(noise_levels, ent_mcd, "s--", label="MC-Dropout")
    ax.plot(noise_levels, ent_bbb, "^-", label="Bayes by Backprop")
    ax.set_xlabel("Gaussian noise std ($\\sigma$)"); ax.set_ylabel("Mean predictive entropy")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    names = ["MAP", "MC-Dropout", "BBB"]
    vals = [ood_ent["map"], ood_ent["mcd"], ood_ent["bbb"]]
    ax.bar(names, vals, color=["#55A868", "#C44E52", "#4C72B0"])
    ax.set_ylabel("Mean predictive entropy"); ax.set_title("Pure-noise OOD inputs", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Predictive uncertainty under distribution shift", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close()


def fig_small_data(small_res, full_res, path):
    methods = ["MAP (L2)", "MC-Dropout", "Bayes by Backprop"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    acc_full = [full_res["map"]["acc"], full_res["mcd"]["acc"], full_res["bbb"]["acc"]]
    acc_small = [small_res["map"]["acc"], small_res["mcd"]["acc"], small_res["bbb"]["acc"]]
    ece_full = [full_res["map"]["ece"], full_res["mcd"]["ece"], full_res["bbb"]["ece"]]
    ece_small = [small_res["map"]["ece"], small_res["mcd"]["ece"], small_res["bbb"]["ece"]]
    x = np.arange(3); w = 0.36
    axes[0].bar(x - w / 2, np.array(acc_full) * 100, w, label="54k train", color="#4C72B0")
    axes[0].bar(x + w / 2, np.array(acc_small) * 100, w, label="5k train", color="#DD8452")
    axes[0].set_xticks(x); axes[0].set_xticklabels(methods, fontsize=8)
    axes[0].set_ylabel("Test accuracy (%)"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(x - w / 2, np.array(ece_full) * 100, w, label="54k train", color="#4C72B0")
    axes[1].bar(x + w / 2, np.array(ece_small) * 100, w, label="5k train", color="#DD8452")
    axes[1].set_xticks(x); axes[1].set_xticklabels(methods, fontsize=8)
    axes[1].set_ylabel("ECE (%)"); axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Effect of training-set size on accuracy and calibration", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close()


# ----------------------------------------------------------------------------
# Hyper-parameter selection on the validation set
# ----------------------------------------------------------------------------
N_TRAIN = 54000   # set in main(); used for KL scaling in the sweep


def val_accuracy(model, Xva, yva, kind, t=20):
    if kind == "map":
        p = predict_map(model, Xva)
    elif kind == "mcd":
        p = predict_map(model, Xva)   # deterministic eval (dropout off) for selection
    else:
        p = predict_bbb(model, Xva, T=t)
    return (p.argmax(1) == yva).float().mean().item()


def sweep_hyperparams(loader_tr, Xva, yva):
    """Select hyper-parameters on the validation set (test set untouched)."""
    print("[sweep] validation-based hyper-parameter selection", flush=True)
    best = {"map_l2": 1e-4, "mcd_p": 0.5, "bbb_prior": 1.0}

    scores = []
    for l2 in [1e-4, 1e-3]:
        net = MAPNet(dropout=0.0)
        train_map(net, loader_tr, SWEEP_EPOCHS, l2=l2)
        acc = val_accuracy(net, Xva, yva, "map")
        scores.append((acc, l2))
        print(f"[sweep] MAP l2={l2}: val_acc={acc:.4f}", flush=True)
    best["map_l2"] = max(scores)[1]

    scores = []
    for p in [0.3, 0.5]:
        net = MAPNet(dropout=p)
        train_map(net, loader_tr, SWEEP_EPOCHS)
        acc = val_accuracy(net, Xva, yva, "mcd")
        scores.append((acc, p))
        print(f"[sweep] MC-Dropout p={p}: val_acc={acc:.4f}", flush=True)
    best["mcd_p"] = max(scores)[1]

    scores = []
    for prior in [0.1, 0.3, 1.0]:
        net = BBBNet(prior_std=prior)
        train_bbb(net, loader_tr, SWEEP_EPOCHS, n_train=N_TRAIN)
        acc = val_accuracy(net, Xva, yva, "bbb")
        scores.append((acc, prior))
        print(f"[sweep] BBB prior_std={prior}: val_acc={acc:.4f}", flush=True)
    best["bbb_prior"] = max(scores)[1]
    print(f"[sweep] chosen: {best}", flush=True)
    return best


def sweep_bbb_prior_small(loader_sm, Xva, yva, n_small):
    """Select BBB prior for the small-data regime on the validation set."""
    scores = []
    for prior in [0.1, 0.3, 1.0]:
        net = BBBNet(prior_std=prior)
        train_bbb(net, loader_sm, SWEEP_EPOCHS, n_train=n_small)
        acc = val_accuracy(net, Xva, yva, "bbb")
        scores.append((acc, prior))
        print(f"[sweep] BBB (small) prior_std={prior}: val_acc={acc:.4f}", flush=True)
    return max(scores)[1]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    global N_TRAIN
    (Xtrn, ytrn), (Xva, yva), (Xte, yte), (Xsm, ysm) = load_mnist()
    N_TRAIN = Xtrn.size(0)
    print(f"[data] train={Xtrn.size(0)} val={Xva.size(0)} test={Xte.size(0)} small={Xsm.size(0)}", flush=True)

    loader_tr = make_loader(Xtrn, ytrn)
    loader_sm = make_loader(Xsm, ysm)
    results = {}

    # --- 0. hyper-parameter selection (validation set only) ---
    hp = sweep_hyperparams(loader_tr, Xva, yva)
    hp_small_prior = sweep_bbb_prior_small(loader_sm, Xva, yva, Xsm.size(0))
    results["hyperparams"] = {"full": hp, "bbb_prior_small": hp_small_prior}

    # --- 1. full-data training ---
    print("[train] MAP", flush=True)
    map_net = MAPNet(dropout=0.0)
    hist_map = train_map(map_net, loader_tr, EPOCHS_FULL, l2=hp["map_l2"])
    torch.save(map_net.state_dict(), os.path.join(OUT_DIR, "model_map.pt"))

    print("[train] MC-Dropout", flush=True)
    mcd_net = MAPNet(dropout=hp["mcd_p"])
    hist_mcd = train_map(mcd_net, loader_tr, EPOCHS_FULL)
    torch.save(mcd_net.state_dict(), os.path.join(OUT_DIR, "model_mcdropout.pt"))

    print("[train] Bayes by Backprop", flush=True)
    bbb_net = BBBNet(prior_std=hp["bbb_prior"])
    hist_bbb = train_bbb(bbb_net, loader_tr, EPOCHS_FULL, n_train=Xtrn.size(0))
    torch.save(bbb_net.state_dict(), os.path.join(OUT_DIR, "model_bbb.pt"))

    # --- 1b. full-data evaluation on clean test ---
    print("[eval] clean test", flush=True)
    p_map = predict_map(map_net, Xte)
    p_mcd = predict_mc_dropout(mcd_net, Xte)
    p_bbb = predict_bbb(bbb_net, Xte)
    full_res = {"map": metrics(p_map, yte), "mcd": metrics(p_mcd, yte), "bbb": metrics(p_bbb, yte)}
    results["full"] = full_res
    rel_map = reliability_data(p_map, yte)
    rel_mcd = reliability_data(p_mcd, yte)
    rel_bbb = reliability_data(p_bbb, yte)

    fig_train_curves(hist_map, hist_mcd, hist_bbb, os.path.join(OUT_DIR, "fig1_train_curves.png"))
    fig_reliability(rel_map, rel_mcd, rel_bbb, os.path.join(OUT_DIR, "fig2_reliability.png"))

    # --- 2. corruption robustness + OOD ---
    print("[eval] corruption", flush=True)
    noise_levels = [0.0, 0.3, 0.5]
    ent = {"map": [], "mcd": [], "bbb": []}
    acc_corr = {"map": [], "mcd": [], "bbb": []}
    for sigma in noise_levels:
        if sigma == 0.0:
            Xc = Xte
        else:
            Xc = torch.clamp(Xte + torch.randn_like(Xte) * sigma, 0.0, 1.0)
        pm = predict_map(map_net, Xc)
        pmc = predict_mc_dropout(mcd_net, Xc)
        pb = predict_bbb(bbb_net, Xc)
        for key, p in [("map", pm), ("mcd", pmc), ("bbb", pb)]:
            m = metrics(p, yte)
            ent[key].append(m["entropy"])
            acc_corr[key].append(m["acc"])
    results["corruption"] = {"noise_levels": noise_levels, "entropy": ent, "acc": acc_corr}

    # OOD: pure Gaussian noise
    X_ood = torch.rand_like(Xte)  # uniform noise images (far from digit manifold)
    ood_ent = {
        "map": metrics(predict_map(map_net, X_ood), yte)["entropy"],
        "mcd": metrics(predict_mc_dropout(mcd_net, X_ood), yte)["entropy"],
        "bbb": metrics(predict_bbb(bbb_net, X_ood), yte)["entropy"],
    }
    results["ood"] = ood_ent
    fig_entropy_corruption(noise_levels, ent["map"], ent["mcd"], ent["bbb"], ood_ent,
                           os.path.join(OUT_DIR, "fig3_entropy_corruption.png"))

    # --- 3. small-data adaptation ---
    print("[train] small data (5k)", flush=True)
    map_small = MAPNet(dropout=0.0)
    train_map(map_small, loader_sm, EPOCHS_SMALL, l2=hp["map_l2"])
    mcd_small = MAPNet(dropout=hp["mcd_p"])
    train_map(mcd_small, loader_sm, EPOCHS_SMALL)
    bbb_small = BBBNet(prior_std=hp_small_prior)
    train_bbb(bbb_small, loader_sm, EPOCHS_SMALL, n_train=Xsm.size(0))

    print("[eval] small-data models on clean test", flush=True)
    small_res = {
        "map": metrics(predict_map(map_small, Xte), yte),
        "mcd": metrics(predict_mc_dropout(mcd_small, Xte), yte),
        "bbb": metrics(predict_bbb(bbb_small, Xte), yte),
    }
    results["small"] = small_res
    fig_small_data(small_res, full_res, os.path.join(OUT_DIR, "fig4_small_data.png"))

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] total {time.time() - t0:.1f}s; results in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
