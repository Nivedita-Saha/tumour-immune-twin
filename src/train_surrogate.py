"""
Train the neural surrogate (step A3.1).

The surrogate is the digital twin itself: a fast, differentiable stand-in for
the mechanistic model, learned from simulated data.

Three design decisions
----------------------

1. The network sees the patient's parameters, not just their state.

   Two patients in an identical state receiving an identical dose evolve
   differently if their tumour growth rates differ. So "state plus dose
   predicts next state" is not a well-defined function across a diverse
   cohort. Conditioning on the eight varied parameters makes it well
   defined, and matches what a digital twin actually is: a model calibrated
   to a specific individual rather than a population average.

2. The network predicts the CHANGE in state, not the next state.

   Changes are small and centred near zero, which is much easier for a
   network to fit than absolute values. It also means predicting "nothing
   happens" is free, rather than something the network must learn.

3. Inputs and outputs are normalised per dimension.

   Step A2.2 measured that the drug moves about ten times faster per step
   than the tumour. Without normalisation the network would minimise total
   error by fitting the drug accurately and treating the tumour as noise,
   which is exactly backwards.

Normalisation statistics are computed from the TRAINING split only. Using
the full dataset would leak information about test patients into training.

Outputs
-------
    models/surrogate_mlp.pt          trained weights and normalisation stats
    figures/a3_1_training.png        loss curves and error breakdown
    figures/a3_1_rollout_check.png   a first look at multi-step behaviour

Run with:
    python src/train_surrogate.py
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = "data"
MODEL_DIR = "models"
FIGURE_DIR = "figures"

RANDOM_SEED = 0

# Training settings
HIDDEN_SIZE = 128
N_HIDDEN_LAYERS = 3
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
MAX_EPOCHS = 120
PATIENCE = 12          # stop if validation has not improved for this many epochs

STATE_LABELS = ["N healthy", "T tumour", "I immune", "u drug"]


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class Surrogate(nn.Module):
    """
    A plain feedforward network mapping

        [state (4), dose (1), patient parameters (8)]  ->  change in state (4)

    Kept deliberately simple. A small MLP is the honest first attempt, and it
    gives the Neural ODE in step A3.2 something concrete to be measured
    against. Reaching for the complicated model first would leave no way to
    show the complexity was warranted.
    """

    def __init__(self, n_inputs, n_outputs, hidden=HIDDEN_SIZE, n_layers=N_HIDDEN_LAYERS):
        super().__init__()

        layers = [nn.Linear(n_inputs, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, n_outputs)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Normaliser:
    """
    Standardises each dimension to roughly zero mean and unit spread.

    Statistics come from the training split only. A small floor is applied to
    the standard deviation so a near-constant dimension cannot blow up when
    divided.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = torch.clamp(std, min=1e-8)

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, z):
        return z * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d):
        return cls(d["mean"], d["std"])


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_data():
    """Load transitions and splits, and assemble network inputs and targets."""
    trans_path = os.path.join(DATA_DIR, "transitions.npz")
    splits_path = os.path.join(DATA_DIR, "splits.npz")

    for path in (trans_path, splits_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run generate_data.py and make_splits.py first."
            )

    trans = np.load(trans_path, allow_pickle=True)
    splits = np.load(splits_path, allow_pickle=True)

    states = trans["states"].astype(np.float32)
    doses = trans["doses"].astype(np.float32).reshape(-1, 1)
    next_states = trans["next_states"].astype(np.float32)
    params = trans["patient_params"].astype(np.float32)

    # Inputs: current state, applied dose, and who this patient is.
    X = np.concatenate([states, doses, params], axis=1)
    # Targets: how much the state changes over one timestep.
    Y = next_states - states

    masks = {
        "train": splits["train_mask"],
        "val": splits["val_mask"],
        "test": splits["test_mask"],
    }

    return X, Y, masks, trans


def make_tensors(X, Y, masks):
    """Split into tensors per set."""
    out = {}
    for name, mask in masks.items():
        out[name] = (
            torch.from_numpy(X[mask]),
            torch.from_numpy(Y[mask]),
        )
    return out


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(model, x_norm, y_norm, data, device):
    """
    Train with early stopping on validation loss.

    Early stopping keeps the weights from the best validation epoch rather
    than the last one, which guards against the network slowly memorising
    the training patients.
    """
    X_train, Y_train = data["train"]
    X_val, Y_val = data["val"]

    Xtr = x_norm.encode(X_train).to(device)
    Ytr = y_norm.encode(Y_train).to(device)
    Xva = x_norm.encode(X_val).to(device)
    Yva = y_norm.encode(Y_val).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=0.5, patience=5
    )
    loss_fn = nn.MSELoss()

    n_train = len(Xtr)
    train_history, val_history = [], []

    best_val = float("inf")
    best_weights = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        permutation = torch.randperm(n_train, device=device)
        running_loss = 0.0

        for start in range(0, n_train, BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            xb, yb = Xtr[idx], Ytr[idx]

            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * len(idx)

        train_loss = running_loss / n_train

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), Yva).item()

        scheduler.step(val_loss)
        train_history.append(train_loss)
        val_history.append(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == MAX_EPOCHS - 1:
            print(f"  epoch {epoch:>3}   train {train_loss:.6f}   val {val_loss:.6f}")

        if epochs_without_improvement >= PATIENCE:
            print(f"  early stop at epoch {epoch}, no improvement for {PATIENCE} epochs")
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return train_history, val_history, best_val


def evaluate(model, x_norm, y_norm, data, device, split="test"):
    """
    Report one-step error in raw units, per state dimension.

    Errors are reported denormalised so they are interpretable: an error in
    tumour population means something, an error in normalised units does not.
    """
    X, Y = data[split]
    model.eval()
    with torch.no_grad():
        pred_norm = model(x_norm.encode(X).to(device)).cpu()
    pred = y_norm.decode(pred_norm)

    abs_error = (pred - Y).abs()
    mae = abs_error.mean(dim=0)
    rmse = ((pred - Y) ** 2).mean(dim=0).sqrt()

    # Relative to how much each quantity actually moves per step, which is
    # the meaningful comparison.
    typical_change = Y.abs().mean(dim=0)
    relative = mae / torch.clamp(typical_change, min=1e-8)

    return mae, rmse, typical_change, relative


# ----------------------------------------------------------------------
# Rollout sanity check
# ----------------------------------------------------------------------

def rollout_check(model, x_norm, y_norm, trans, splits_path, device, n_patients=3):
    """
    Roll the surrogate forward many steps and compare against the truth.

    One-step error being small does not guarantee a usable model. Errors
    compound: a small mistake changes the next input, which produces a
    slightly larger mistake, and so on. Since the controller will act over a
    full horizon, rollout behaviour matters far more than one-step accuracy.

    This is a first look. Step A3.3 does the full validation.
    """
    splits = np.load(splits_path, allow_pickle=True)
    test_patients = splits["test_patients"]

    patient_ids = trans["patient_ids"]
    states = trans["states"].astype(np.float32)
    doses = trans["doses"].astype(np.float32)
    next_states = trans["next_states"].astype(np.float32)
    params = trans["patient_params"].astype(np.float32)

    results = []
    model.eval()

    for pid in test_patients[:n_patients]:
        # Take one trajectory belonging to this patient.
        idx = np.flatnonzero(patient_ids == pid)
        if len(idx) < 200:
            continue
        idx = idx[:200]   # one trajectory of 200 steps

        true_states = np.concatenate([states[idx], next_states[idx][-1:]], axis=0)
        applied = doses[idx]
        patient_params = params[idx[0]]

        # Roll forward using only the surrogate's own predictions.
        state = torch.from_numpy(states[idx[0]].copy())
        predicted = [state.numpy().copy()]

        with torch.no_grad():
            for step in range(len(idx)):
                x = torch.cat([
                    state,
                    torch.tensor([applied[step]], dtype=torch.float32),
                    torch.from_numpy(patient_params),
                ]).unsqueeze(0)

                delta_norm = model(x_norm.encode(x).to(device)).cpu()
                delta = y_norm.decode(delta_norm).squeeze(0)

                state = state + delta
                predicted.append(state.numpy().copy())

        predicted = np.array(predicted)
        error = np.abs(predicted - true_states)

        results.append({
            "patient": int(pid),
            "true": true_states,
            "pred": predicted,
            "final_error": error[-1],
            "mean_error": error.mean(axis=0),
        })

    return results


def plot_training(train_history, val_history, mae, typical_change, rollouts):
    """Loss curves, per-dimension error, and a rollout overlay."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(train_history, label="train")
    ax.plot(val_history, label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("normalised MSE")
    ax.set_title("Training curves")
    ax.legend()

    ax = axes[1]
    x = np.arange(len(STATE_LABELS))
    width = 0.38
    ax.bar(x - width / 2, mae.numpy(), width, label="one-step error", color="#a4243b")
    ax.bar(x + width / 2, typical_change.numpy(), width,
           label="typical step change", color="#0f4c5c")
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_LABELS, rotation=20, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_title("Error against how much each quantity moves")
    ax.legend(fontsize=8)

    ax = axes[2]
    if rollouts:
        r = rollouts[0]
        steps = np.arange(len(r["true"]))
        ax.plot(steps, r["true"][:, 1], color="#0f4c5c", linewidth=2, label="true tumour")
        ax.plot(steps, r["pred"][:, 1], color="#a4243b", linestyle="--",
                linewidth=1.6, label="surrogate")
        ax.plot(steps, r["true"][:, 0], color="#1b7a5a", linewidth=2, label="true healthy")
        ax.plot(steps, r["pred"][:, 0], color="#c78c3c", linestyle="--",
                linewidth=1.6, label="surrogate")
        ax.set_xlabel("step"); ax.set_ylabel("population")
        ax.set_title(f"Rollout, unseen patient {r['patient']}")
        ax.legend(fontsize=7)

    fig.suptitle("Neural surrogate training (A3.1)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, "a3_1_training.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # A small MLP trains quickly on CPU, and CPU avoids backend quirks.
    device = torch.device("cpu")

    print("Loading data...")
    X, Y, masks, trans = load_data()
    data = make_tensors(X, Y, masks)

    n_inputs, n_outputs = X.shape[1], Y.shape[1]
    print(f"  inputs  {n_inputs}  (4 state + 1 dose + 8 patient parameters)")
    print(f"  outputs {n_outputs}  (change in each state variable)")
    for name in ("train", "val", "test"):
        print(f"  {name:<6} {len(data[name][0]):>8,} transitions")
    print()

    # Normalisation statistics from the training split only.
    X_train, Y_train = data["train"]
    x_norm = Normaliser(X_train.mean(dim=0), X_train.std(dim=0))
    y_norm = Normaliser(Y_train.mean(dim=0), Y_train.std(dim=0))

    model = Surrogate(n_inputs, n_outputs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {N_HIDDEN_LAYERS} hidden layers of {HIDDEN_SIZE}, "
          f"{n_params:,} weights\n")

    print("Training...")
    start = time.time()
    train_history, val_history, best_val = train(model, x_norm, y_norm, data, device)
    elapsed = time.time() - start
    print(f"  done in {elapsed:.1f} s, best validation loss {best_val:.6f}\n")

    mae, rmse, typical_change, relative = evaluate(
        model, x_norm, y_norm, data, device, split="test"
    )

    print("One-step accuracy on unseen test patients:")
    print(f"{'quantity':<12} {'MAE':>10} {'RMSE':>10} {'typical move':>14} {'error/move':>12}")
    print("-" * 62)
    for i, label in enumerate(STATE_LABELS):
        print(f"{label:<12} {mae[i]:>10.6f} {rmse[i]:>10.6f} "
              f"{typical_change[i]:>14.6f} {relative[i]:>11.2%}")
    print()

    print("Rollout check on unseen patients...")
    rollouts = rollout_check(
        model, x_norm, y_norm, trans,
        os.path.join(DATA_DIR, "splits.npz"), device
    )
    for r in rollouts:
        print(f"  patient {r['patient']:>3}   "
              f"mean error over 200 steps: "
              f"N {r['mean_error'][0]:.4f}  T {r['mean_error'][1]:.4f}  "
              f"I {r['mean_error'][2]:.4f}")
    print()

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "surrogate_mlp.pt")
    torch.save({
        "model_state": model.state_dict(),
        "x_norm": x_norm.state_dict(),
        "y_norm": y_norm.state_dict(),
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
        "hidden": HIDDEN_SIZE,
        "n_layers": N_HIDDEN_LAYERS,
        "seed": RANDOM_SEED,
    }, model_path)

    fig_path = plot_training(train_history, val_history, mae, typical_change, rollouts)

    print("Saved:")
    print(f"  {model_path}")
    print(f"  {fig_path}")
    print()
    print("One-step accuracy is necessary but not sufficient. Step A3.3 tests")
    print("whether errors stay bounded over a full horizon, which is what the")
    print("controller will depend on.")
