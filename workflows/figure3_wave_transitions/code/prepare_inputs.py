"""Generate all six Figure 3 panels from CPC scores and HMM posteriors."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
from scipy.stats import false_discovery_control

from summary_statistics import mean_valid, paired_pvalues, step_summary, unit

ROOT = Path(__file__).resolve().parents[1]
CORE = slice(100, 1100)
LAGS = np.arange(-5, 6)
KEYS = np.array([12 * i + j for i in range(12) for j in range(12) if i != j])


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def grouped(values, keys):
    """Average events within each directed pair and lag, retaining missing values."""
    valid = np.isfinite(values)
    sums = np.zeros((144, len(LAGS)), dtype=values.dtype)
    counts = np.zeros((144, len(LAGS)), int)
    np.add.at(sums, keys, np.where(valid, values, 0))
    np.add.at(counts, keys, valid)
    return np.divide(
        sums, counts, out=np.full(sums.shape, np.nan, dtype=sums.dtype), where=counts > 0
    )


def checked_run(scores, posterior, run):
    c = np.asarray(scores[run, :, :30], dtype=np.complex128)
    alpha = np.asarray(posterior[run])
    if not np.isfinite(c).all() or not np.isfinite(alpha).all():
        raise ValueError(f"Non-finite scores or posteriors in run {run}")
    if np.any(alpha < 0) or not np.allclose(alpha.sum(1), 1, atol=1e-4):
        raise ValueError(f"Posteriors are not normalized probabilities in run {run}")
    return c, alpha.argmax(1)


def window_events(labels):
    times = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    times = times[(times >= 105) & (times < 1095)]
    return times, labels[times - 1] * 12 + labels[times]


def select_cpcs(scores, posterior, participants):
    """Select one CPC per directed transition using REST1_LR only."""
    sumsq, effect = np.zeros(30), np.zeros((144, 30), complex)
    counts = np.zeros(144, np.int64)
    samples = 0
    for participant in range(participants):
        c, y = checked_run(scores, posterior, 4 * participant)
        sample = c[100:1100:4]
        sumsq += (abs(sample) ** 2).sum(0)
        samples += len(sample)
        times, keys = window_events(y)
        np.add.at(effect, keys, c[times + 2] - c[times - 2])
        counts += np.bincount(keys, minlength=144)
    rms = np.sqrt(sumsq / samples)
    mean_change = effect / np.maximum(counts[:, None], 1)
    standardized = np.divide(mean_change, rms, out=np.zeros_like(effect), where=rms > 0)
    chosen = abs(standardized).argmax(1)
    chosen[counts == 0] = 0
    return chosen, counts


def observed_probability(c, labels, chosen):
    """P(next state | current state, current-frame CPC phase), without shift controls."""
    bins = np.floor(((np.angle(c) + np.pi) % (2 * np.pi)) * 24 / (2 * np.pi)).astype(int)
    key = ((np.arange(30)[:, None] * 12 + labels[:-1]) * 24 + bins[:-1].T) * 12 + labels[1:]
    # A phase at zero amplitude is undefined, not phase zero.
    valid = abs(c[:-1].T) > 1e-12
    joint = np.bincount(key[valid], minlength=30 * 12 * 24 * 12).reshape(30, 12, 24, 12)
    numerator = joint[chosen[KEYS], KEYS // 12, :, KEYS % 12]
    denominator = joint.sum(3)[chosen[KEYS], KEYS // 12]
    return np.divide(numerator, denominator, out=np.full((132, 24), np.nan), where=denominator > 0)


def save_examples(scores, posterior, anchors_file, output):
    """Keep illustration anchors; locate the nearest actual switch if a fit changes."""
    with anchors_file.open() as handle:
        anchors = list(csv.DictReader(handle))
    if len(anchors) != 3:
        raise ValueError("Supply exactly three illustration anchors (one for each CPC1–3).")
    coefficients = np.full((3, 17, 30), np.nan + 0j)
    labels = np.zeros((3, 17), int)
    records = []
    for i, anchor in enumerate(anchors):
        run, frame, component = [int(anchor[key]) for key in ["run_index", "frame", "CPC"]]
        if run < 0 or run >= len(scores) or run % 4 not in (2, 3) or component != i + 1:
            raise ValueError("Example anchors must name evaluation runs and CPC1, CPC2, CPC3.")
        if frame < 101 or frame >= 1100:
            raise ValueError("Example anchor frames must be within 101–1099.")
        c, y = checked_run(scores, posterior, run)
        events = np.flatnonzero(y[1:] != y[:-1]) + 1
        events = events[(events >= 101) & (events < 1100)]
        selected = int(events[np.argmin(abs(events - frame))]) if len(events) else -1
        if selected >= 0:
            # Preserve the stored score precision used by the manuscript examples.
            coefficients[i] = scores[run, selected - 8 : selected + 9, :30]
            labels[i] = y[selected - 8 : selected + 9] + 1
        records.append(
            dict(
                example=i + 1,
                run_index=run,
                anchor_frame=frame,
                transition_frame=selected,
                CPC=component,
                source=int(y[selected - 1]) + 1 if selected >= 0 else 0,
                target=int(y[selected]) + 1 if selected >= 0 else 0,
            )
        )
    np.savez_compressed(
        output / "examples.npz",
        coefficients=coefficients,
        labels=labels,
        valid=np.array([r["transition_frame"] >= 0 for r in records]),
    )
    write_csv(output / "individual_event_selection.csv", records)


def prepare(args):
    scores = np.load(args.scores, mmap_mode="r")
    posterior = np.load(args.posterior, mmap_mode="r")
    if (
        scores.ndim != 3
        or scores.shape[1] != 1200
        or scores.shape[2] < 30
        or len(scores) == 0
        or len(scores) % 4
        or not np.iscomplexobj(scores)
        or posterior.shape != (len(scores), 1200, 12)
    ):
        raise ValueError(
            "Expected complex (4*N,1200,>=30) scores and matching (4*N,1200,12) posteriors."
        )
    participants = len(scores) // 4
    args.output.mkdir(parents=True, exist_ok=False)
    # Validate and generate illustrations before the more expensive cohort pass.
    save_examples(scores, posterior, args.examples, args.output)
    chosen, training_counts = select_cpcs(scores, posterior, participants)
    print("Training-only CPC selection complete.", flush=True)
    distance = np.empty((participants, 2, 999))
    switches = np.empty(distance.shape, bool)
    transition_counts = np.zeros((12, 12), np.int64)
    window_counts = np.zeros(144, np.int64)
    amplitude = np.full((participants, 132, 2), np.nan)
    phase = np.full(amplitude.shape, np.nan + 0j)
    probability_sum, probability_n = np.zeros((132, 24)), np.zeros((132, 24), int)
    for participant in range(participants):
        run_amplitude, run_phase, run_probability = [], [], []
        for acquisition in range(2):
            run = participant * 4 + 2 + acquisition
            c, y = checked_run(scores, posterior, run)
            steps = np.linalg.norm(np.diff(c, axis=0), axis=1)
            distance[participant, acquisition] = steps[100:1099]
            switches[participant, acquisition] = y[100:1099] != y[101:1100]
            yc = y[CORE]
            transition_counts += np.bincount(yc[:-1] * 12 + yc[1:], minlength=144).reshape(12, 12)
            times, keys = window_events(y)
            window_counts += np.bincount(keys, minlength=144)
            windows = c[times[:, None] + LAGS, chosen[keys, None]]
            scale = abs(c[CORE]).mean(0)[chosen[keys], None]
            normalized = np.divide(
                abs(windows),
                scale,
                out=np.full(windows.shape, np.nan, dtype=float),
                where=scale > 0,
            )
            run_amplitude.append(grouped(normalized, keys)[KEYS])
            run_phase.append(grouped(unit(windows), keys)[KEYS])
            run_probability.append(observed_probability(c[CORE], yc, chosen))
        participant_amplitude = mean_valid(np.stack(run_amplitude))
        participant_phase = mean_valid(np.stack(run_phase))
        amplitude[participant] = np.stack(
            [
                mean_valid(participant_amplitude[:, period], axis=1)
                for period in [slice(0, 5), slice(5, 10)]
            ],
            axis=-1,
        )
        phase[participant] = unit(
            np.stack(
                [
                    mean_valid(participant_phase[:, period], axis=1)
                    for period in [slice(0, 5), slice(5, 10)]
                ],
                axis=-1,
            )
        )
        probability = mean_valid(np.stack(run_probability))
        supported = np.isfinite(probability)
        probability_sum += np.where(supported, probability, 0)
        probability_n += supported
        if participant % 250 == 0:
            print(f"Evaluation: {participant + 1}/{participants} participants.", flush=True)
    for values in [amplitude, phase]:
        values[~np.isfinite(values).all(axis=-1)] = np.nan
    pa, pz, na, nz = paired_pvalues(
        amplitude[:, :, 1] - amplitude[:, :, 0],
        phase[:, :, 1] - phase[:, :, 0],
        args.permutations,
        args.seed,
    )
    qa, qz = np.split(false_discovery_control(np.r_[pa, pz], method="by"), 2)
    mean_phase = mean_valid(phase)
    phase_angle = np.rad2deg(np.angle(mean_phase))
    phase_angle[abs(mean_phase) <= 1e-12] = np.nan
    row_counts = transition_counts.sum(1, keepdims=True)
    transition_probability = np.divide(
        transition_counts, row_counts, out=np.full((12, 12), np.nan), where=row_counts > 0
    )
    np.savez_compressed(
        args.output / "transition_summary.npz",
        transition_probability=transition_probability,
        transition_counts=transition_counts,
        transition_keys=KEYS,
        CPC=chosen[KEYS] + 1,
        training_counts=training_counts[KEYS],
        selection_fallback=training_counts[KEYS] == 0,
        amplitude_mean=mean_valid(amplitude),
        phase_mean_degrees=phase_angle,
        amplitude_participants=na,
        phase_participants=nz,
        amplitude_p=pa,
        phase_p=pz,
        amplitude_q=qa,
        phase_q=qz,
        amplitude_significant=qa < 0.05,
        phase_significant=qz < 0.05,
        window_counts=window_counts[KEYS],
        before_TR=LAGS[:5],
        after_TR=LAGS[5:10],
        participants=participants,
        permutations=args.permutations,
        permutation_seed=args.seed,
    )
    probability = np.divide(
        probability_sum, probability_n, out=np.full((132, 24), np.nan), where=probability_n > 0
    )
    np.savez_compressed(
        args.output / "phase_conditioned_probability.npz",
        probability_percent=100 * probability,
        transition_keys=KEYS,
        CPC=chosen[KEYS] + 1,
        phase_degrees=np.linspace(-172.5, 172.5, 24),
        participants=probability_n,
    )
    distance, switches = distance.reshape(participants, -1), switches.reshape(participants, -1)
    edges = np.arange(0, np.ceil(max(float(distance.max()), 3.0) / 0.2) * 0.2 + 0.3, 0.2)
    histograms = np.array(
        [
            [np.histogram(v[m == condition], bins=edges)[0] for condition in [False, True]]
            for v, m in zip(distance, switches)
        ],
        dtype=np.int32,
    )
    means = np.column_stack(
        [
            mean_valid(np.where(switches == condition, distance, np.nan), axis=1)
            for condition in [False, True]
        ]
    )
    np.savez_compressed(
        args.output / "displacement_plot_inputs.npz",
        bin_edges=edges,
        participant_histograms=histograms,
        participant_means=means,
    )
    rows = [
        step_summary(distance, switches == condition, name)
        for condition, name in [(False, "nontransition"), (True, "transition")]
    ]
    write_csv(args.output / "violin_statistics.csv", rows)
    coverage = [
        dict(
            source=int(key // 12 + 1),
            target=int(key % 12 + 1),
            CPC=int(chosen[key] + 1),
            training_events=int(training_counts[key]),
            fallback=bool(training_counts[key] == 0),
            events=int(transition_counts[key // 12, key % 12]),
            complete_events=int(window_counts[key]),
        )
        for key in KEYS
    ]
    write_csv(args.output / "transition_coverage.csv", coverage)
    print(f"Prepared six-panel Figure 3 inputs for {participants} participants.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New source-data directory")
    parser.add_argument("--examples", type=Path, default=ROOT / "example_anchors.csv")
    parser.add_argument("--permutations", type=int, default=19999)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260920,
        help="Pre/post permutation seed, not an HMM fitting seed",
    )
    arguments = parser.parse_args()
    if arguments.permutations < 1:
        parser.error("--permutations must be positive")
    prepare(arguments)
