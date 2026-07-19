# Fire and Smoke Alarm Transfer-Learning Design

**Date:** 2026-07-18

**Status:** Approved design, pending implementation plan

**Target:** Earshot demo detector trained on Windows and deployed for inference on Windows or Raspberry Pi

## Objective

Add a small, high-recall `fire_smoke_alarm` detector on top of Earshot's pinned
YAMNet model. YAMNet remains frozen and supplies one 1,024-value embedding per
audio window. A logistic-regression head learns the distinction between the
project's alarm and non-alarm recordings. The exported head runs with NumPy, so
the Raspberry Pi runtime does not gain a TensorFlow or scikit-learn dependency.

The detector intentionally combines smoke-detector and building fire-alarm
sounds into one high-urgency event. The demo prioritizes catching every alarm
in the evaluation corpus over minimizing false alerts.

This feature is not a certified fire, smoke, accessibility, or life-safety
system and must not replace approved alarms or emergency procedures.

## Current State and Dataset

Earshot currently checks YAMNet's raw `Fire alarm` and
`Smoke detector, smoke alarm` scores against fixed thresholds. Its `teach`
feature stores normalized example embeddings and performs nearest-neighbor
cosine matching; it does not fit a classifier or learn from negative examples.

The initial local corpus lives under `ml/data/alarm_demo/` and currently has:

- 7 `alarm` WAV files, covering smoke-detector and fire-alarm sounds.
- 10 `not_alarm` WAV files, covering speech, music, applause, cheering, and a
  barking dog.
- Valid 16-bit PCM WAV containers at 44.1 or 48 kHz; the existing loader can
  downmix stereo and resample to 16 kHz.
- Direct YAMNet alarm peaks of 0.90-1.00 on every positive file and 0.00 on the
  current negative files.

The corpus is suitable for a demo, but it is small and its negatives do not yet
cover difficult alarm-like beeps such as phones, timers, appliance chimes, and
doorbells. Metrics must be described as corpus results, not general real-world
accuracy. The WAVs and generated models are local artifacts and must not be
committed by default.

## Chosen Approach

Use YAMNet as a frozen feature extractor and fit a balanced logistic-regression
classifier on its 1,024-dimensional embeddings.

Rejected alternatives:

- Expanding `teach` prototypes cannot use negative examples or learn a
  discriminative boundary.
- A neural-network head adds overfitting and deployment risk without enough
  data to justify the extra capacity.
- Fine-tuning YAMNet requires much more data and a pinned TensorFlow training
  environment, and risks degrading its pretrained representation.
- Mapping generic `Alarm`, `Beep`, or `Buzzer` classes directly would create
  avoidable false positives.

## Components and Boundaries

### Dataset and collection module

A focused dataset module owns paths, WAV validation, collision-safe copying,
microphone recording, and corpus enumeration. It does not know how embeddings
or classifiers work.

Default layout:

```text
ml/data/alarm_demo/
|-- alarm/
|-- not_alarm/
`-- manifest.json
```

Collection copies source files instead of moving or editing them. It accepts
uncompressed 8-, 16-, or 32-bit PCM WAV input supported by Earshot's loader.
It rejects unsupported containers, unreadable audio, non-finite decoded data,
and clips shorter than one model window. Name collisions receive a stable
content-hash suffix rather than overwriting an existing file.

The manifest records each relative path, label, `source_group`, and optional
positive time segments. Collection
uses one source group per imported or recorded source unless the caller passes
`--source-group` to associate related edits or repeated captures. Exact
duplicates are detected from the SHA-256 digest of decoded, mono, resampled
audio and share one source group; duplicate content is not counted twice.
Manually placed WAVs missing from the manifest receive stable per-file groups
in memory and a prominent report warning to review related-source grouping;
training does not rewrite the corpus or manifest.

### Training module

A training-only module owns preprocessing, augmentation, grouped
cross-validation, logistic-regression fitting, threshold selection, and report
generation. It imports scikit-learn only when a training command runs.

The training extra is installed on Windows with:

```powershell
python -m pip install -e ".[test,train]"
```

The ordinary runtime dependencies remain NumPy, sounddevice, and the existing
LiteRT/TFLite interpreter.

### Runtime model module

A small runtime module validates and loads the generated artifact, standardizes
a YAMNet embedding, computes a numerically stable sigmoid score, and exposes
artifact metadata. It depends only on NumPy and shared artifact utilities.

### Engine integration

`EarshotML` continues to run YAMNet once per window. When a trained alarm head
is loaded, the same embedding feeds the new classifier. Existing pretrained
events other than fire/smoke and all taught events retain their current paths.
The trained alarm uses its own rolling evidence gate before entering the
existing debounce/event delivery path.

## CLI and User Flow

Add these commands:

```powershell
earshot collect alarm alarm1.wav alarm2.wav
earshot collect alarm --record 5 --seconds 5 --device 1
earshot collect not_alarm timer.wav phone.wav
earshot collect not_alarm --record 10 --seconds 5 --device 1
earshot train-alarm
earshot evaluate-alarm
```

All three commands accept an optional `--data-dir`. Collection accepts optional
`--source-group`. Training accepts an optional `--output` and deterministic
`--seed`. Evaluation accepts optional `--model` and `--data-dir` paths.

`collect` permits WAV imports, microphone recordings, or both in one command.
It prints every stored path and a final count. Recorded clips are written as
mono, 16 kHz, float-decoded/16-bit PCM WAV files through an atomic temporary
file and rename.

`train-alarm` validates the full corpus, extracts embeddings, performs grouped
cross-validation, selects a threshold, trains the final head on all recordings,
atomically writes the model artifact, and writes
`ml/models/fire_smoke_alarm_report.json`.

`evaluate-alarm` runs a named corpus through a trained artifact and reports
per-file decisions plus aggregate clip recall, triggered-negative count,
false triggers per audio minute, and latency-to-trigger where applicable. An
evaluation against training data is explicitly labeled as in-sample; the
cross-validation report remains the primary estimate for the initial corpus.

`top5` retains its existing five raw YAMNet classes and adds the current trained
`fire_smoke_alarm` score when a head is loaded.

## Preprocessing and Sampling

All files use the existing Earshot path: decode, downmix, resample to 16 kHz,
then generate 15,600-sample windows with an 8,000-sample hop.

Positive WAVs are required to contain only the target alarm plus silence unless
the manifest supplies alarm-bearing time segments. A window is eligible only
when at least half of it overlaps a supplied segment; without segments, the
whole file is the declared positive range. To remove silence, calculate every
window's RMS and the positive clip's 95th-percentile window RMS, then require an
RMS of at least `max(1e-4, 0.05 * clip_p95_rms)`. Raw YAMNet class scores are
reported as a content-audit aid but never decide which labeled positive windows
are trainable. This allows the new head to learn alarm variants YAMNet currently
scores poorly. Negative-file windows, including quiet room audio and silence,
remain eligible. The report records kept and discarded windows and their
selection reasons per file. A file with no usable positive windows fails
validation with its path in the error.

No recording may dominate because it is long. Each file contributes at most 40
deterministically, evenly spaced eligible windows. Shorter files contribute all
eligible windows. Each retained window receives an inverse-retained-window-count
sample weight for its source recording, and logistic regression also uses
balanced class weights.

Only training folds receive augmentation. For every retained positive training
window, create two additional seeded variants: one with gain sampled uniformly
from 0.35-1.0, and one with the same gain range plus a randomly selected
non-silent negative-training window mixed at a signal-to-noise ratio sampled
uniformly from 8-20 dB. Compute
`target_noise_rms = signal_rms / (10 ** (snr_db / 20))`, then multiply the
selected noise by `target_noise_rms / noise_rms`; retry another negative window
when its RMS is below `1e-6`. Validation recordings and their windows are never
used as augmentation sources for their fold. Audio is clipped safely to
`[-1, 1]` after mixing. The default random seed is 0. The original plus two
descendants divide their parent's sample weight equally, so augmentation does
not triple a positive recording's influence. Fold models and the final all-data
model both use augmentation under these rules.

## Cross-Validation and Threshold Selection

Use five stratified folds grouped by manifest `source_group`, not pathname.
Every related recording, window, and augmentation stays in the same fold. The
command requires at least five usable source groups in each class so every fold
has both labels.

Within each fold:

1. Fit feature normalization using only training embeddings.
2. Fit scikit-learn logistic regression with `C=1.0`, `solver="liblinear"`,
   `class_weight="balanced"`, `max_iter=2000`, the recording sample weights,
   and the configured seed, using only training groups.
3. Score untouched validation groups.
4. Simulate the production rolling evidence gate per recording.

Pool the out-of-fold predictions. Evaluate every unique finite out-of-fold
score plus 0 and 1 as a threshold candidate. A viable candidate must trigger
every positive source group, trigger no more than 20% of negative source
groups, and produce no more than 0.5 debounced false triggers per negative
audio minute. Among viable candidates, choose the highest threshold. Because
trigger decisions are monotonic with the threshold, this preserves the
required corpus recall while minimizing or tying false triggers. If no
candidate satisfies both recall and false-alert ceilings, training writes a
diagnostic report but does not replace the last known-good model artifact.
Temporal-gate and debounce state reset at each file boundary; the rate is the
sum of emitted negative-file events divided by total negative audio duration.

Finally, fit normalization and logistic weights on all usable recordings. Run
that final head over the full corpus and find the highest in-sample threshold
that still triggers every positive source group. Use the lower of that value
and the cross-validated threshold as the deployment threshold, then re-evaluate
the out-of-fold predictions at that deployment threshold. The artifact is
eligible for installation only if the deployment threshold still satisfies the
out-of-fold false-alert ceilings and the final head's full-corpus in-sample run
satisfies the same positive-recall and negative false-alert ceilings. The
report presents out-of-fold estimates and final-model in-sample checks
separately; it never describes the in-sample check as unbiased test
performance. It also records both candidate thresholds, the deployed value,
seed, folds, corpus file hashes, sample counts, metrics, and warnings about the
small or weakly varied corpus.

## Runtime Temporal Decision

The trained detector requires at least two qualifying windows among the most
recent eight half-overlapping windows, approximately four seconds. The windows
do not need to be consecutive, allowing pulsed alarm patterns to contain silent
gaps. Once fired, the event uses the existing debounce interval.

The event payload is:

```json
{
  "label": "fire_smoke_alarm",
  "urgency": "high",
  "confidence": 0.91,
  "source": "trained",
  "timestamp": 1752969600.0
}
```

The confidence is the trained head's sigmoid score, not a calibrated probability.

## Artifact Contract

The default output is `ml/models/fire_smoke_alarm_head.npz`, resolved under the
same model directory as YAMNet and overridable with
`EARSHOT_ALARM_MODEL_PATH`. The artifact
contains only non-pickle arrays and scalar metadata:

- Schema/magic identifier and version.
- Event label and urgency.
- Feature dimension.
- Normalization mean and scale.
- Logistic weights and bias.
- Selected decision threshold.
- Rolling gate count and window length.
- YAMNet model and class-map SHA-256 digests.

Loading rejects missing keys, object arrays, unexpected shapes or dtypes,
non-finite values, invalid thresholds/gate settings, unsupported schema
versions, and YAMNet digest mismatches. Writes use a temporary file, flush and
filesystem sync where supported, then atomic replacement. A failed write must
preserve any previous known-good artifact.

If no trained artifact exists, Earshot preserves its existing pretrained
`fire_alarm` and `smoke_alarm` behavior. If a valid head exists, it suppresses
those two generic mappings and emits only the shared trained event. If an
artifact exists but is corrupt or incompatible, startup fails with a concise
diagnostic instead of silently falling back.

## Error Handling

Commands fail before mutation when:

- A class directory is missing or empty.
- Either class has fewer than five usable source groups.
- A WAV is invalid, unsupported, too short, or decodes to invalid samples.
- A positive file has no active alarm windows after preprocessing.
- Embedding extraction fails or yields an invalid shape/value.
- Cross-validation cannot produce both labels in every training fold.
- No threshold satisfies both 100% positive source-group recall and the
  false-alert ceilings.
- No finite model or threshold can be produced.
- The model artifact cannot be installed atomically.

Errors identify the operation and affected path without dumping tracebacks for
ordinary user mistakes. Unexpected programmer errors remain visible in tests.

## Testing Strategy

### Unit tests

- Dataset layout, validation, safe copying, collision handling, and recording
  output format.
- Activity filtering, per-file caps, deterministic sampling, augmentation
  isolation/weight conservation, manifest source grouping, exact-duplicate
  handling, and grouped split leakage prevention.
- Deterministic classifier fitting through a fake embedding extractor.
- Threshold selection with 100% positive source-group recall, mandatory
  false-alert ceilings, and rejection of an always-positive classifier.
- Artifact round trip, schema validation, digest mismatch, malformed/non-finite
  arrays, and atomic-write rollback.
- Stable sigmoid inference and the two-of-eight rolling evidence gate.
- Engine behavior with no head, a valid head, and an invalid head, including
  suppression of duplicate generic fire/smoke events.
- CLI success and concise failure paths for `collect`, `train-alarm`, and
  `evaluate-alarm`.

### Integration and regression tests

- Preserve all existing unit and real-model tests.
- Mark corpus evaluation as integration because the local WAVs are not checked
  into source control.
- Train on the current 17-file corpus and verify the generated artifact loads
  and produces finite scores.
- Require 100% out-of-fold positive source-group recall, at most 20% triggered
  negative source groups, and at most 0.5 debounced false triggers per negative
  audio minute for the initial demo corpus.
- Report, but do not conceal or reinterpret, every triggered negative file and
  false-trigger rate.
- Run direct-file evaluation and a later microphone acceptance pass on Windows
  device 1 and the target Pi microphone.

## Repository and Deployment Hygiene

Add `ml/data/` to `.gitignore` so locally sourced audio is not accidentally
committed. Keep generated `.npz` files and `ml/models/*_report.json` ignored.
Documentation
explains how to copy `fire_smoke_alarm_head.npz` to the Pi's configured model
directory and verify its YAMNet digest before running.

The implementation must preserve all user WAVs. Collection copies inputs;
training and evaluation are read-only with respect to the corpus.

## Acceptance Criteria

- The new CLI workflow works from the documented Windows project-local virtual
  environment.
- The current 7 positive and 10 negative WAVs pass dataset validation.
- Training completes deterministically and emits a validated, atomically
  installed NumPy artifact plus a JSON report.
- Cross-validation triggers all positive corpus source groups while satisfying
  the mandatory negative-group and false-triggers-per-minute ceilings.
- `top5` exposes the trained score and `run` emits a debounced
  `fire_smoke_alarm` event using two-of-eight temporal evidence.
- Runtime on the Pi requires no scikit-learn or TensorFlow training package.
- Existing pretrained non-alarm events, teach mode, callback behavior, and CLI
  commands remain compatible.
- All existing and new non-integration tests pass; the local corpus integration
  evaluation passes its high-recall requirement.
- README documentation includes collection, training, evaluation, deployment,
  troubleshooting, metric interpretation, and the life-safety disclaimer.
