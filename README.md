# Spike sorting with Kilosort4 for AIND ephys pipeline
## aind-ephys-spikesort-kilosort4


### Description

This capsule is designed to spike sort ephys data using [Kilosort4](https://github.com/MouseLand/Kilosort/) for the AIND pipeline.

This capsule spike sorts preprocessed ephys stream and applies a minimal curation to:

- remove empty units
- remove excess spikes (falling beyond the end of the recording)


### Inputs

The `data/` folder must include the output of the [aind-ephys-preprocessing](https://github.com/AllenNeuralDynamics/aind-ephys-preprocessing), containing 
the `data/preprocessed_{recording_name}` folder.

### Parameters

The `code/run` script takes the following arguments:

```bash
  --raise-if-fails      Whether to raise an error in case of failure or continue. Default True (raise)
  --skip-motion-correction
                        Whether to skip Kilosort4 motion correction. Default: True
  --min-drift-channels MIN_DRIFT_CHANNELS
                        Minimum number of channels to enable Kilosort4 motion correction. Default is 96.
  --n-jobs N_JOBS       Number of jobs to use for parallel processing. Default is -1 (all available cores). 
                        It can also be a float between 0 and 1 to use a fraction of available cores
  --params-file PARAMS_FILE
                        Optional json file with parameters
  --params-str PARAMS_STR
                        Optional json string with parameters

```

A list of spike sorting parameters can be found in the `code/params.json`:

```json
{
    "job_kwargs": {
        "chunk_duration": "1s",
        "progress_bar": false
    },
    "sorter": {
        "batch_size": 60000,
        "nblocks": 5,
        "Th_universal": 9,
        "Th_learned": 8,
        "do_CAR": true,
        "invert_sign": false,
        "nt": 61,
        "shift": null,
        "scale": null,
        "artifact_threshold": null,
        "nskip": 25,
        "whitening_range": 32,
        "highpass_cutoff": 300,
        "binning_depth": 5,
        "sig_interp": 20,
        "drift_smoothing": [0.5, 0.5, 0.5],
        "nt0min": null,
        "dmin": null,
        "dminx": 32,
        "min_template_size": 10,
        "template_sizes": 5,
        "nearest_chans": 10,
        "nearest_templates": 100,
        "max_channel_distance": null,
        "templates_from_data": true,
        "n_templates": 6,
        "n_pcs": 6,
        "Th_single_ch": 6,
        "acg_threshold": 0.2,
        "ccg_threshold": 0.25,
        "cluster_downsampling": 20,
        "cluster_pcs": 64,
        "x_centers": null,
        "duplicate_spike_ms": 0.25,
        "scaleproc": null,
        "save_preprocessed_copy": false,
        "torch_device": "auto",
        "bad_channels": null,
        "clear_cache": false,
        "save_extra_vars": false,
        "do_correction": true,
        "keep_good_only": false,
        "skip_kilosort_preprocessing": false,
        "use_binary_file": null,
        "delete_recording_dat": true
    }
}
```

### Output

The output of this capsule is the following:

- `results/spikesorted_{recording_name}` folder, containing the spike sorted data saved by SpikeInterface and the spike sorting log
- `results/data_process_spikesorting_{recording_name}.json` file, a JSON file containing a `DataProcess` object from the [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package.

### Fixed-template score diagnostic

The owned calibration capsule also supports an explicit diagnostic entry point:

```bash
./run --score-diagnostic
```

This mode expects the exact 1,200-second denoised input mounted as
`full96_om1_probec_1200s` and the corresponding raw/ground-truth asset mounted as
`probec_recording1_3`. It learns templates once from the denoised input, then
replays the same templates, whitening, drift, and channel selection on matched
raw and denoised samples.

Score replay covers `Th_learned` 8, 9, 10, and 10.75. Full event lineage is
recorded at the endpoint thresholds 8 and 10.75 across these stages:

1. learned detection template;
2. final graph clustering;
3. CCG-guided cluster merging;
4. duplicate removal.

The lineage CSVs report stage totals, per-GT-unit counts, per-cluster properties,
adjacent-stage deltas, status transitions, detection-template-to-cluster
transitions, and extraction-score distributions. Compressed NPZ files preserve
the complete event identity and status arrays. Event status uses the benchmark
matching settings: 0.4 ms tolerance, Hungarian unit matching, and minimum
agreement 0.2.

`fp_matched_cluster` means an event in a sorter cluster matched to an injected
GT unit but not temporally matched to that unit. It is not necessarily electrical
noise; native biological spikes can receive this benchmark label. The lineage is
a controlled fixed-denoised-template experiment, not an independently relearned
production sort for each input domain.

To characterize the native raw baseline on the same 1,200-second interval, run:

```bash
./run --score-diagnostic-raw-native
```

This mode requires only the `probec_recording1_3` raw/GT mount. It derives the
bad-channel mask, whitening, drift estimate, and learned templates from raw
voltage at the default `Th_universal=9`, `Th_learned=8`, then runs the identical
lineage analysis at learned threshold 8. Its final sort is checked against the
raw template-learning sort for exact times and cluster partition. The event-level
archive retains the fields needed for baseline peel/score analysis in compact
form; every stage, unit, cluster, transition, and score summary remains available
as CSV. Comparing this result with the denoised threshold-8 lineage gives the
native default-baseline comparison; the fixed-template raw replay above answers
a different controlled transfer question.

### Experimental target-decoy FDR gate

The capsule includes a GT-blind experimental matcher:

```bash
./run --target-decoy-diagnostic
```

The registered experiment uses the same first 1,200 seconds of raw and Full96
omission1 voltage. Each domain independently learns its native channel mask,
whitening, drift, and templates at the default thresholds. During matching
pursuit, positive learned-template local maxima are targets and sign-reversed
local maxima are decoys. For every batch and peel, the matcher chooses the
smallest threshold at or above 8 satisfying the knockoff-plus estimate

```text
(1 + decoys above threshold) / targets above threshold <= 0.05.
```

The `+1` makes the rule conservative and requires at least 20 accepted events
within a batch/peel at 5% FDR. Ground truth is not used for matching decisions;
it is applied only afterward for the same lineage evaluation used by the
baseline diagnostics. The result includes the selected threshold, target/decoy
counts, estimated FDR, and sign-balance ratio for every batch and peel.

This is a research prototype, not a production recommendation. Its main
assumption is that negative signed template maxima form an exchangeable null for
positive residual artifacts after native preprocessing. That assumption must be
checked on raw and denoised data. Short smoke runs may use `--duration-s` and
`--domain`; the registered comparison remains fixed at 1,200 seconds, both
domains, target FDR 5%, and threshold floor 8.

### Experimental local joint amplitude refit

The capsule also includes a GT-blind matching-pursuit modification that targets
structured subtraction residuals directly:

```bash
./run --joint-refit-diagnostic
```

Detection remains at the native default learned threshold 8. After each original
greedy subtraction peel, every newly detected event defines a local block
containing itself and all positive-amplitude events whose center lies within the
full `ctc` lag support. The matcher solves the nonnegative local problem

```text
minimize 0.5 * a.T * G * a - b.T * a, subject to a >= 0,
```

where `G[i, j] = ctc[template_i, template_j, time_i - time_j + nt]` and `b` is
recovered exactly from the current residual projection plus the block's current
contribution. Amplitude deltas rebuild both the voltage residual and all template
projections before the next peel. Overlapping blocks are processed in ascending
sample order as sequential block-coordinate updates. Identical time-template
detections share one amplitude variable.

The solver uses an active-set NNLS update, a machine-precision eigenspace rank
cutoff, and no ridge or ground-truth-derived parameter. Per-batch/per-peel output
reports greedy and additional refit energy reduction, overlap-block sizes, Gram
conditioning, rank and convergence safeguards, amplitude changes, zeroed events,
late-peel counts, and runtime. Ground truth is applied only after extraction by
the same lineage evaluator used for the baseline diagnostics.

Each run also executes unchanged Kilosort 4.1.7 with the identical native
preprocessing, learned `ops`, templates, binary recording, and batch boundaries.
The stage tables contain both arms, and `joint_refit_peel_comparison.csv` reports
exact same-template event deltas at each peel and at-or-after each peel. Use
`--skip-baseline-control` only for mechanism-development runs where that paired
control has already been established.

Short smoke runs may use `--duration-s` and `--domain`. The registered comparison
is fixed at 1,200 seconds, both raw-native and denoised-native domains, threshold
8, one-hop full-`ctc` support, and the solver policy above.

