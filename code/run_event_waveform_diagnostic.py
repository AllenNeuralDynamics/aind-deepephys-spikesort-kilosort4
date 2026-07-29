"""Compare archived native-denoised TP and FP events on paired source voltage."""
from __future__ import annotations

import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from event_waveform import paired_waveform_cosine, rank_auc, waveform_shape_metrics


SELECTION_PATH = Path(__file__).with_name("event_waveform_selection.json")
WINDOW_SAMPLES = 61
EDGE_SAMPLES = 10
EXPECTED_SAMPLING_FREQUENCY = 30_000.0
RAW_ASSET = "8046af5a-6e53-420e-9e28-52bd54514342"
DENOISED_ASSET = "e7134769-b8c9-437a-ba0d-f5bd9ee0078b"
COLORS = {"tp": "#167D63", "fp": "#C6403D"}


def baseline_correct(waveforms: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (waveforms[..., :EDGE_SAMPLES], waveforms[..., -EDGE_SAMPLES:]), axis=-1
    )
    return waveforms - np.median(edges, axis=-1, keepdims=True)


def extract_waveforms(recording, events: list[dict], unit_lookup: dict[int, dict]) -> np.ndarray:
    half = WINDOW_SAMPLES // 2
    output = np.empty(
        (len(events), len(next(iter(unit_lookup.values()))["channel_ids"]), WINDOW_SAMPLES),
        dtype=np.float32,
    )
    for index, event in enumerate(events):
        time_sample = event["time_sample"]
        if time_sample < half or time_sample + half >= recording.get_num_samples():
            raise ValueError(f"event {event['event_id']} lies outside waveform bounds")
        output[index] = recording.get_traces(
            start_frame=time_sample - half,
            end_frame=time_sample + half + 1,
            channel_ids=unit_lookup[event["gt_unit_id"]]["channel_ids"],
            return_scaled=True,
        ).T
        if (index + 1) % 100 == 0:
            print(f"extracted {index + 1}/{len(events)} waveforms", flush=True)
    return baseline_correct(output)


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.9)),
    }


def add_shape_metrics(
    event_rows: list[dict],
    waveforms: dict[str, np.ndarray],
    unit_lookup: dict[int, dict],
) -> tuple[pd.DataFrame, dict[tuple[int, str], np.ndarray]]:
    frame = pd.DataFrame(event_rows)
    frame["dominant_template_event"] = [
        event["detection_template"]
        == unit_lookup[event["gt_unit_id"]]["dominant_detection_template"]
        for event in event_rows
    ]
    references: dict[tuple[int, str], np.ndarray] = {}
    for unit_id in unit_lookup:
        unit = frame.gt_unit_id == unit_id
        reference_events = (
            unit
            & (frame.event_class == "tp")
            & frame.reference_tp
            & frame.dominant_template_event
        )
        if reference_events.sum() < 2:
            raise ValueError(f"GT unit {unit_id} has too few reference TP events")
        for domain in ("raw", "denoised"):
            reference = np.median(waveforms[domain][reference_events], axis=0)
            references[(unit_id, domain)] = reference
            cosine, scale, residual = waveform_shape_metrics(
                waveforms[domain][unit], reference
            )
            indices = frame.index[unit]
            frame.loc[indices, f"{domain}_tp_reference_cosine"] = cosine
            frame.loc[indices, f"{domain}_tp_reference_scale"] = scale
            frame.loc[indices, f"{domain}_tp_reference_residual_fraction"] = residual
            local = waveforms[domain][unit]
            frame.loc[indices, f"{domain}_peak_channel_ptp"] = np.ptp(
                local[:, 0, :], axis=1
            )
            edges = np.concatenate(
                (local[..., :EDGE_SAMPLES], local[..., -EDGE_SAMPLES:]), axis=-1
            )
            noise = np.median(np.abs(edges), axis=(1, 2)) / 0.67448975
            frame.loc[indices, f"{domain}_local_snr"] = (
                np.ptp(local[:, 0, :], axis=1) / noise.clip(1e-6)
            )
            trough_offset = (
                np.argmin(local[:, 0, :], axis=1)
                - np.argmin(reference[0])
            )
            frame.loc[indices, f"{domain}_trough_offset_samples"] = trough_offset
            frame.loc[indices, f"{domain}_trough_offset_abs_samples"] = np.abs(
                trough_offset
            )
    frame["raw_denoised_cosine"] = paired_waveform_cosine(
        waveforms["raw"], waveforms["denoised"]
    )
    return frame, references


def summarize_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = (
        "detection_score",
        "detection_peel",
        "raw_tp_reference_cosine",
        "denoised_tp_reference_cosine",
        "raw_tp_reference_residual_fraction",
        "denoised_tp_reference_residual_fraction",
        "raw_peak_channel_ptp",
        "denoised_peak_channel_ptp",
        "raw_local_snr",
        "denoised_local_snr",
        "raw_trough_offset_abs_samples",
        "denoised_trough_offset_abs_samples",
        "raw_denoised_cosine",
    )
    summary_rows = []
    auc_rows = []
    evaluation = ~frame.reference_tp
    subsets = {
        "all_templates": np.ones(len(frame), dtype=bool),
        "dominant_template": frame.dominant_template_event.to_numpy(),
    }
    for subset_name, subset in subsets.items():
        for unit_id, unit_frame in frame[evaluation & subset].groupby("gt_unit_id"):
            for event_class, class_frame in unit_frame.groupby("event_class"):
                for metric in metrics:
                    values = class_frame[metric].to_numpy(dtype=float)
                    row = {
                        "gt_unit_id": int(unit_id),
                        "subset": subset_name,
                        "event_class": event_class,
                        "metric": metric,
                        "events": int(values.size),
                        **quantiles(values),
                    }
                    summary_rows.append(row)
            labels = (unit_frame.event_class == "tp").to_numpy()
            if labels.all() or (~labels).all():
                continue
            for metric in metrics:
                values = unit_frame[metric].to_numpy(dtype=float)
                lower_for_tp = (
                    "residual_fraction" in metric
                    or "trough_offset_abs_samples" in metric
                    or metric == "detection_peel"
                )
                direction = -1 if lower_for_tp else 1
                auc_rows.append(
                    {
                        "gt_unit_id": int(unit_id),
                        "subset": subset_name,
                        "metric": metric,
                        "tp_higher_direction": direction,
                        "tp_vs_fp_auc": rank_auc(labels, direction * values),
                        "tp_events": int(labels.sum()),
                        "fp_events": int((~labels).sum()),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(auc_rows)


def representative_examples(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    evaluation = frame[~frame.reference_tp]
    for unit_id, unit_frame in evaluation.groupby("gt_unit_id"):
        for event_class, class_frame in unit_frame.groupby("event_class"):
            dominant = class_frame[class_frame.dominant_template_event]
            if not dominant.empty:
                class_frame = dominant
            target = class_frame.denoised_tp_reference_cosine.median()
            selected = class_frame.iloc[
                np.argmin(np.abs(class_frame.denoised_tp_reference_cosine - target))
            ]
            row = selected.to_dict()
            row["waveform_index"] = int(selected.name)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_trace_band(axis, time_ms, values, event_class: str) -> None:
    median = np.median(values, axis=0)
    low, high = np.quantile(values, (0.1, 0.9), axis=0)
    axis.plot(time_ms, median, color=COLORS[event_class], label=event_class.upper())
    axis.fill_between(time_ms, low, high, color=COLORS[event_class], alpha=0.16)


def plot_overview(
    frame: pd.DataFrame,
    waveforms: dict[str, np.ndarray],
    units: list[dict],
    sampling_frequency: float,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_ms = (
        (np.arange(WINDOW_SAMPLES) - WINDOW_SAMPLES // 2)
        / sampling_frequency
        * 1000
    )
    figure, axes = plt.subplots(len(units), 4, figsize=(15, 3.25 * len(units)))
    for row, unit in enumerate(units):
        unit_id = unit["gt_unit_id"]
        selected = (frame.gt_unit_id == unit_id) & ~frame.reference_tp
        for column, domain in enumerate(("raw", "denoised")):
            axis = axes[row, column]
            for event_class in ("tp", "fp"):
                mask = selected & (frame.event_class == event_class)
                if mask.any():
                    plot_trace_band(
                        axis, time_ms, waveforms[domain][mask, 0, :], event_class
                    )
            axis.axvline(0, color="#777777", linewidth=0.7)
            axis.set_title(f"GT {unit_id}: {domain} peak channel")
            axis.set_xlabel("Time (ms)")
            axis.set_ylabel("Voltage")
            axis.legend(frameon=False)

        axis = axes[row, 2]
        for event_class in ("tp", "fp"):
            mask = selected & (frame.event_class == event_class)
            if mask.any():
                axis.scatter(
                    frame.loc[mask, "detection_peel"],
                    frame.loc[mask, "detection_score"],
                    s=16,
                    alpha=0.65,
                    color=COLORS[event_class],
                    label=event_class.upper(),
                )
        axis.set_title("Kilosort evidence")
        axis.set_xlabel("Peel")
        axis.set_ylabel("Detection score")

        axis = axes[row, 3]
        box_values = []
        labels = []
        colors = []
        for domain in ("raw", "denoised"):
            for event_class in ("tp", "fp"):
                mask = selected & (frame.event_class == event_class)
                if mask.any():
                    box_values.append(
                        frame.loc[mask, f"{domain}_tp_reference_cosine"].to_numpy()
                    )
                    labels.append(f"{domain[:3]}\n{event_class.upper()}")
                    colors.append(COLORS[event_class])
        boxes = axis.boxplot(box_values, labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        axis.set_ylim(-1.05, 1.05)
        axis.set_title("Cosine to held-out TP reference")
        axis.set_ylabel("Centered cosine")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_examples(
    examples: pd.DataFrame,
    waveforms: dict[str, np.ndarray],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    units = sorted(examples.gt_unit_id.unique())
    figure, axes = plt.subplots(len(units), 4, figsize=(14, 2.8 * len(units)))
    if len(units) == 1:
        axes = axes[None, :]
    for row, unit_id in enumerate(units):
        for column, (domain, event_class) in enumerate(
            (("raw", "tp"), ("raw", "fp"), ("denoised", "tp"), ("denoised", "fp"))
        ):
            axis = axes[row, column]
            match = examples[
                (examples.gt_unit_id == unit_id)
                & (examples.event_class == event_class)
            ]
            if match.empty:
                axis.axis("off")
                continue
            event_index = int(match.iloc[0].waveform_index)
            waveform = waveforms[domain][event_index]
            limit = float(np.max(np.abs(waveform)))
            axis.imshow(
                waveform,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                extent=(-1, 1, waveform.shape[0] - 0.5, -0.5),
            )
            axis.axvline(0, color="black", linewidth=0.6)
            axis.set_title(
                f"GT {unit_id} {domain} {event_class.upper()}\n"
                f"score {match.iloc[0].detection_score:.1f}, peel {int(match.iloc[0].detection_peel)}"
            )
            axis.set_xlabel("Time (ms)")
            axis.set_ylabel("Local channel rank")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    import spikeinterface as si

    import run_score_diagnostic as diagnostic

    started = time.perf_counter()
    expected_versions = {"spikeinterface": "0.104.7"}
    versions = {name: version(name) for name in expected_versions}
    if versions != expected_versions:
        raise RuntimeError(f"unexpected environment: {versions} != {expected_versions}")
    selection = json.loads(SELECTION_PATH.read_text())
    events = selection["events"]
    units = selection["units"]
    raw_path, denoised_path, _ = diagnostic.discover_inputs()
    raw = si.read_zarr(raw_path)
    denoised = si.load(denoised_path)
    if raw.get_num_segments() != 1 or denoised.get_num_segments() != 1:
        raise ValueError("waveform inputs must each have one segment")
    if not np.array_equal(raw.channel_ids, denoised.channel_ids):
        raise ValueError("raw and denoised channel identities differ")
    if raw.get_num_channels() != 384 or denoised.get_num_channels() != 384:
        raise ValueError("selection requires the original 384-channel recordings")
    if not raw.has_scaleable_traces() or not denoised.has_scaleable_traces():
        raise ValueError("source recordings must expose scaled voltage traces")
    if not np.isclose(raw.get_sampling_frequency(), EXPECTED_SAMPLING_FREQUENCY):
        raise ValueError(f"unexpected raw sampling frequency: {raw.get_sampling_frequency()}")
    if not np.isclose(
        denoised.get_sampling_frequency(), EXPECTED_SAMPLING_FREQUENCY
    ):
        raise ValueError(
            f"unexpected denoised sampling frequency: {denoised.get_sampling_frequency()}"
        )
    end_frame = int(
        round(diagnostic.EXPECTED_DURATION_S * EXPECTED_SAMPLING_FREQUENCY)
    )
    if raw.get_num_samples() < end_frame:
        raise ValueError("raw recording is shorter than the analysis interval")
    raw = raw.frame_slice(0, end_frame)
    if denoised.get_num_samples() != end_frame:
        raise ValueError("denoised recording is not exactly 1200 seconds")
    if max(event["time_sample"] for event in events) + WINDOW_SAMPLES // 2 >= end_frame:
        raise ValueError("selected event lies outside the analysis interval")

    union_indices = sorted(
        {index for unit in units for index in unit["channel_indices"]}
    )
    union_ids = raw.channel_ids[union_indices]
    raw_preprocessed, denoised_preprocessed, kept_ids, removed_ids = (
        diagnostic.matched_preprocessing(raw, denoised)
    )
    missing_ids = [channel_id for channel_id in union_ids if channel_id not in kept_ids]
    if missing_ids:
        raise ValueError(f"selected local channels were removed as bad: {missing_ids}")
    raw_local = raw_preprocessed.select_channels(union_ids)
    denoised_local = denoised_preprocessed.select_channels(union_ids)
    unit_lookup = {}
    for unit in units:
        unit_lookup[unit["gt_unit_id"]] = {
            **unit,
            "channel_ids": raw.channel_ids[unit["channel_indices"]].tolist(),
        }

    print("extracting raw source waveforms", flush=True)
    waveforms = {"raw": extract_waveforms(raw_local, events, unit_lookup)}
    print("extracting denoised source waveforms", flush=True)
    waveforms["denoised"] = extract_waveforms(
        denoised_local, events, unit_lookup
    )
    frame, references = add_shape_metrics(events, waveforms, unit_lookup)
    summary, auc = summarize_metrics(frame)
    examples = representative_examples(frame)

    diagnostic.RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_csv(diagnostic.RESULTS / "event_waveform_event_metrics.csv", index=False)
    summary.to_csv(diagnostic.RESULTS / "event_waveform_summary.csv", index=False)
    auc.to_csv(diagnostic.RESULTS / "event_waveform_auc.csv", index=False)
    examples.to_csv(diagnostic.RESULTS / "event_waveform_examples.csv", index=False)
    reference_arrays = {
        f"unit_{unit_id}_{domain}": value
        for (unit_id, domain), value in references.items()
    }
    np.savez_compressed(
        diagnostic.RESULTS / "event_waveforms.npz",
        raw=waveforms["raw"],
        denoised=waveforms["denoised"],
        event_id=frame.event_id.to_numpy(dtype=np.int64),
        gt_unit_id=frame.gt_unit_id.to_numpy(dtype=np.int32),
        event_class=frame.event_class.to_numpy(dtype="U2"),
        **reference_arrays,
    )
    plot_overview(
        frame,
        waveforms,
        units,
        raw.get_sampling_frequency(),
        diagnostic.RESULTS / "event_waveform_overview.png",
    )
    plot_examples(
        examples,
        waveforms,
        diagnostic.RESULTS / "event_waveform_examples.png",
    )
    manifest = {
        **{key: value for key, value in selection.items() if key != "events"},
        "units": list(unit_lookup.values()),
        "raw_asset": RAW_ASSET,
        "denoised_asset": DENOISED_ASSET,
        "selected_events": len(events),
        "waveform_samples": WINDOW_SAMPLES,
        "waveform_duration_ms": WINDOW_SAMPLES / raw.get_sampling_frequency() * 1000,
        "selected_channel_ids": union_ids.tolist(),
        "removed_channel_ids": removed_ids,
        "preprocessing": (
            "matched source voltage; phase correction when metadata are present; "
            "300 Hz high-pass; denoised-derived bad-channel exclusion; global median "
            "reference; no whitening"
        ),
        "reference_policy": (
            "dominant-template members among half of each stratified TP sample "
            "construct a median reference; remaining TPs and all FPs are held out"
        ),
        "example_policy": (
            "event nearest its held-out class median denoised cosine among dominant-"
            "template events"
        ),
        "spikeinterface_version": versions["spikeinterface"],
        "elapsed_s": time.perf_counter() - started,
    }
    (diagnostic.RESULTS / "event_waveform_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=diagnostic.json_default) + "\n"
    )
    print(json.dumps(manifest, indent=2, default=diagnostic.json_default), flush=True)


if __name__ == "__main__":
    main()