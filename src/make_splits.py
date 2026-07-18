"""
Split the dataset by patient (step A2.3).

Why not split by transition
---------------------------
Shuffling 400,000 transitions and taking 70 percent for training would be a
serious mistake. Consecutive transitions from one patient are nearly
identical, so the same patient would appear in both training and test sets.
The test score would then measure memorisation rather than generalisation,
and would look excellent while the surrogate failed on any new patient.

Splitting at the patient level means the test set contains patients the
surrogate has never seen in any form. That is the honest question: does it
generalise to someone new?

Stratification
--------------
Splits are stratified by untreated outcome, so each split contains a similar
mix of controlled, escape, and host failure patients rather than leaving the
balance to chance.

Outputs
-------
    data/splits.npz             which patient belongs to which split
    figures/a2_3_splits.png     composition of each split

Run with:
    python src/make_splits.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_DIR = "data"
FIGURE_DIR = "figures"

RANDOM_SEED = 123

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
# test takes the remainder


def stratified_patient_split(outcomes, seed=RANDOM_SEED):
    """
    Assign patients to train, validation and test sets.

    Patients are grouped by untreated outcome, and each group is divided
    according to the same proportions. This keeps the outcome mix comparable
    across splits.

    Returns three arrays of patient indices.
    """
    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = [], [], []

    for outcome in np.unique(outcomes):
        members = np.flatnonzero(outcomes == outcome)
        rng.shuffle(members)

        n = len(members)
        n_train = int(round(TRAIN_FRACTION * n))
        n_val = int(round(VAL_FRACTION * n))

        train_ids.append(members[:n_train])
        val_ids.append(members[n_train:n_train + n_val])
        test_ids.append(members[n_train + n_val:])

    return (
        np.sort(np.concatenate(train_ids)),
        np.sort(np.concatenate(val_ids)),
        np.sort(np.concatenate(test_ids)),
    )


def verify_no_overlap(train_ids, val_ids, test_ids, n_patients):
    """
    Confirm the splits are disjoint and complete.

    This check is cheap and catches the exact bug that would invalidate every
    downstream result, so it runs every time rather than being assumed.
    """
    sets = [set(train_ids.tolist()), set(val_ids.tolist()), set(test_ids.tolist())]

    for i, j, name in [(0, 1, "train/val"), (0, 2, "train/test"), (1, 2, "val/test")]:
        shared = sets[i] & sets[j]
        if shared:
            raise AssertionError(f"{name} overlap: {len(shared)} patients in both")

    total = sum(len(s) for s in sets)
    if total != n_patients:
        raise AssertionError(f"splits cover {total} patients, expected {n_patients}")

    return True


def transition_masks(patient_ids, train_ids, val_ids, test_ids):
    """Turn patient-level splits into masks over the transition dataset."""
    return (
        np.isin(patient_ids, train_ids),
        np.isin(patient_ids, val_ids),
        np.isin(patient_ids, test_ids),
    )


def plot_splits(outcomes, train_ids, val_ids, test_ids, counts):
    """Show that the outcome mix is comparable across splits."""
    os.makedirs(FIGURE_DIR, exist_ok=True)

    split_names = ["train", "validation", "test"]
    id_groups = [train_ids, val_ids, test_ids]
    unique_outcomes = sorted(np.unique(outcomes).tolist())

    colours = {
        "controlled": "#1b7a5a",
        "escape": "#a4243b",
        "host failure": "#4a1c2f",
        "intermediate": "#c78c3c",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Patient counts per split, broken down by outcome.
    ax = axes[0]
    bottom = np.zeros(3)
    for outcome in unique_outcomes:
        values = np.array([
            int((outcomes[ids] == outcome).sum()) for ids in id_groups
        ], dtype=float)
        ax.bar(split_names, values, bottom=bottom,
               color=colours.get(outcome, "grey"), label=outcome, alpha=0.9)
        bottom += values
    ax.set_ylabel("patients")
    ax.set_title("Patients per split")
    ax.legend(fontsize=7)

    # Outcome proportions, which is what stratification is meant to equalise.
    ax = axes[1]
    width = 0.25
    x = np.arange(len(unique_outcomes))
    for k, (name, ids) in enumerate(zip(split_names, id_groups)):
        shares = [
            100.0 * (outcomes[ids] == outcome).sum() / len(ids)
            for outcome in unique_outcomes
        ]
        ax.bar(x + (k - 1) * width, shares, width, label=name, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_outcomes, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("percent of split")
    ax.set_title("Outcome mix is comparable")
    ax.legend(fontsize=7)

    # Transition counts, which is what the network actually consumes.
    ax = axes[2]
    ax.bar(split_names, counts, color="#0f4c5c", alpha=0.9)
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("transitions")
    ax.set_title("Transitions per split")

    fig.suptitle("Patient-level data splits (A2.3)", fontsize=13, fontweight="bold")
    fig.tight_layout()

    path = os.path.join(FIGURE_DIR, "a2_3_splits.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    cohort_path = os.path.join(DATA_DIR, "cohort.npz")
    trans_path = os.path.join(DATA_DIR, "transitions.npz")

    for path in (cohort_path, trans_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run cohort.py and generate_data.py first."
            )

    cohort = np.load(cohort_path, allow_pickle=True)
    outcomes = np.array([str(x) for x in cohort["outcomes"]])
    n_patients = len(outcomes)

    transitions = np.load(trans_path, allow_pickle=True)
    patient_ids = transitions["patient_ids"]

    print(f"Splitting {n_patients} patients, stratified by untreated outcome...\n")

    train_ids, val_ids, test_ids = stratified_patient_split(outcomes)
    verify_no_overlap(train_ids, val_ids, test_ids, n_patients)
    print("Verified: no patient appears in more than one split.\n")

    train_mask, val_mask, test_mask = transition_masks(
        patient_ids, train_ids, val_ids, test_ids
    )
    counts = [int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())]

    print(f"{'split':<12} {'patients':>9} {'transitions':>13}")
    print("-" * 36)
    for name, ids, count in zip(
        ["train", "validation", "test"], [train_ids, val_ids, test_ids], counts
    ):
        print(f"{name:<12} {len(ids):>9} {count:>13,}")
    print(f"{'total':<12} {n_patients:>9} {sum(counts):>13,}")
    print()

    print("Outcome mix per split (percent):")
    unique_outcomes = sorted(np.unique(outcomes).tolist())
    header = f"{'outcome':<15}" + "".join(f"{n:>13}" for n in ["train", "validation", "test"])
    print(header)
    print("-" * len(header))
    for outcome in unique_outcomes:
        row = f"{outcome:<15}"
        for ids in (train_ids, val_ids, test_ids):
            share = 100.0 * (outcomes[ids] == outcome).sum() / len(ids)
            row += f"{share:>12.1f}%"
        print(row)
    print()

    os.makedirs(DATA_DIR, exist_ok=True)
    splits_path = os.path.join(DATA_DIR, "splits.npz")
    np.savez_compressed(
        splits_path,
        train_patients=train_ids,
        val_patients=val_ids,
        test_patients=test_ids,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        random_seed=np.array(RANDOM_SEED),
    )

    fig_path = plot_splits(outcomes, train_ids, val_ids, test_ids, counts)

    print("Saved:")
    print(f"  {splits_path}")
    print(f"  {fig_path}")
    print()
    print("Phase 2 complete. The test set contains patients the surrogate will")
    print("never see during training, so its score measures generalisation.")
