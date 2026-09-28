"""Prepare the selected-CPC reference required by aggregate_prepost.py."""

import argparse
from pathlib import Path
import numpy as np


def prepare(profiles, phases):
    """Select each transition's training-chosen CPC from all-component summaries.

    Outputs have (132, 11) transition/time shape; diagonal state pairs are excluded.
    No component is selected using the evaluation summaries.
    """
    keys = np.array([i * 12 + j for i in range(12) for j in range(12) if i != j])
    chosen = profiles["chosen"]
    c = chosen[keys]
    assert chosen.shape == (144,) and np.all((chosen >= 0) & (chosen < 30))

    def select(a):
        return a[keys[:, None], np.arange(11)[None, :], c[:, None]]

    return dict(
        chosen=chosen,
        window_counts=profiles["window_counts"],
        amplitude_reference=select(profiles["amplitude_mean"]),
        phase_reference=select(phases["absolute_phase_mean"]),
        participant_reference=select(profiles["amplitude_participants"]),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["profiles", "phases", "out"]:
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    with np.load(a.profiles) as d, np.load(a.phases) as z:
        result = prepare(d, z)
    with a.out.open("xb") as f:
        np.savez_compressed(f, **result)
