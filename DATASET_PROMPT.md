# Dataset and Analysis Context — Tube-Tilt Experiments

This document provides full context for an AI assistant reviewing this work
alongside Graziano's paper. It describes the experimental task, data
collection, participant selection, statistical approach, and current results.

---

## Theoretical Background

This work is directly motivated by Michael Graziano's **Body Schema** theory
and his work on peripersonal space (PPS) — the region of space immediately
surrounding the body that the brain represents with particular priority. PPS
is not merely spatial; it is affect-laden, threat-sensitive, and face-directed.
Faces that represent social threat or approach signals are hypothesised to
elicit stronger motor engagement with the space they appear to occupy.

The core prediction being tested: **when a face is present and visible, people
will show a directional motor bias — tilting a tube slightly further toward the
face than away from it.** This is a small but systematic effect, measurable
only in aggregate across many participants, and should be modulated by face
salience (eyes open vs. covered, threat level).

---

## The Task

Participants completed a **tube-tilt task** delivered as a web experiment on
their own mobile phone or computer.

On each trial:
- A **profile photograph of a face** appeared on one side of the screen
  (left or right, randomised).
- A virtual **tube** (a rectangle with a dotted centre line) appeared in the
  middle of the screen.
- An **arrow** indicated the direction to tilt (left or right).
- Participants physically tilted their device until they felt the tube would
  "tip" — releasing liquid from the virtual tube.
- The tube rectangle disappeared during the tilt, leaving only the dotted
  centre line as visual feedback.
- The device's tilt angle at release (`endAngle`) was recorded in degrees.

**Towards trials**: the arrow pointed toward the face side.
**Away trials**: the arrow pointed away from the face side.

The task is implicit — participants were not told about the face or asked to
respond to it. They were simply asked to tilt the tube in the indicated
direction until it would tip.

---

## Tube Types

Four tube geometries were used (counterbalanced across trials):

| Index | Width | Height | Critical angle arctan(W/H) |
|-------|-------|--------|---------------------------|
| 0     | 24 px | 130 px | 10.5°                     |
| 1     | 48 px | 130 px | 20.3°                     |
| 2     | 24 px |  65 px | 20.3°                     |
| 3     | 48 px |  65 px | 36.4°                     |

The **critical angle** is the geometric minimum tilt for liquid to reach the
tube rim. Users systematically undershoot the critical angle for harder tubes
(especially tube [3] at 36.4°), averaging only ~20° across all tube types —
an anchoring effect driven by the centre-line visual reference.

Each session contains **32 trials** (4 tube types × 2 tip directions × 4
repetitions per combination, with towards and away interspersed).

The `faceAngle` field stored in the raw JSON (values −9° to +1°) is a
**cosmetic rendering parameter only** — it tilts the tube face graphic to look
toward the top of the tube. It has no relation to the physical tipping
threshold.

---

## Angle Conventions

| Column        | Definition |
|---------------|-----------|
| `end_angle`   | Raw signed angle from the device (°). Negative = tilted left, positive = right. |
| `raw_angle`   | Identical to `end_angle` (preserved for traceability). |
| `tip_direction` | 'left' if `raw_angle < 0`, else 'right'. |
| `angle`       | **Sign-corrected tilt**: `raw_angle` is negated if `tip_direction == 'left'`. Always positive for correct-direction tilts. This is the primary analysis variable. |

**No absolute value is ever taken.** The sign flip is applied once at load
time so that `angle > 0` always means the participant tilted in the direction
of the tip arrow, and `angle < 0` means they tilted the wrong way.

---

## The Outcome Measure: D

For each participant, **D** is computed as:

    D = mean(angle | towards trials) − mean(angle | away trials)

D > 0 means the participant tilted further on towards trials than away trials.
D is computed per participant and then tested across participants using a
one-sample t-test against zero (H₀: D = 0, no directional bias).

D is in degrees. Effect sizes in this paradigm are small (~0.2°) because the
face-induced bias is a subtle motor modulation, not a large explicit response.

---

## Experiment 1 (Exp 1)

**Face**: ID008 — young adult female, photographed in profile, low rated
threat (audience rating 1.75 / 5 on a Likert scale).

**Manipulation**: Within-participant, between-condition.
- **Eyes Open (sighted)**: face visible with eyes open.
- **Eyes Covered (blindfold)**: same face wearing an opaque blindfold.

**Collection**: Crowd-sourced online. Participants recruited via Prolific and
related platforms. Data collected across multiple session groups (identified
by prefix codes such as AT001, AT002, etc.).

**Raw data**: 700 unique participants loaded from 989 JSON files.
37 participants (5.3%) were automatically flagged and removed as bots based on
behavioural criteria (all-zero angles, uniform low-variance angles, wrong
direction tilts, implausibly fast response times, very low RT variability).
This left **663 sessions** entering the pipeline.

**Trials per participant**: 32 (standard); a small number have more due to
duplicate session files that are deduplicated during loading.

**Response time**: Median ~4.2 s per trial (stored in milliseconds as
`latency`). Bot detection partly relies on trials completed in < 1 s.

---

## Experiment 2 (Exp 2)

**Manipulation**: Within-participant, face identity / threat level.

**Faces** (current analysis uses ID015 + ID017 only):
| Face  | Description                                     | Threat | Status           |
|-------|-------------------------------------------------|--------|------------------|
| ID015 | Young adult male, minimal threat cues           | Low    | Active (all runs)|
| ID017 | Adult male, heavy tattoos                       | High   | Active (Run 1+3) |
| ID030 | Adult male, tattoos and visible scarring        | High   | Excluded for now |

Each participant saw one **meek** (ID015) and one **threatening** face across
all 32 trials (16 each).

**Data sources** (all raw files live in `data/tube/raw/exp2/`, never modified):
- **Run 1** (ID015 + ID017): raw JSONs were lost in a data breach. A
  pre-processed CSV (`exp2_threat_run1.csv`, 145 participants, 4,691 trials)
  is the only source. The angle columns in this CSV are used as-is — the
  sign-correction transform was already applied when the CSV was built.
- **Run 2** (ID015 + ID030): 150 participants loaded from raw JSON files.
  Excluded from current analysis — we need more data on the ID015 + ID017
  pairing to confirm the hypothesis before introducing a third face.
- **Run 3** (ID015 + ID017): MTurk collection, March 2026. 500 raw JSON
  files from 252 unique MTurk workers. Severe bot contamination: 77 workers
  submitted multiple sessions (up to 35 from a single worker), nearly all
  with 100% zero angles and sub-second response times.

**Quarantine system**: Because Run 3 introduced `workerId` tracking (via the
`data.session.mturk.workerId` field), the pipeline now classifies raw files
before loading:

```
data/tube/raw/exp2/
    *.json, *.csv           ← raw files (NEVER modified)
    quarantined/            ← copies of files from repeat workers (>1 session per workerId)
    user-data/              ← copies of clean files (pipeline loads from here)
```

`quarantine_workers()` runs at the start of each analysis. It reads every JSON
in the raw directory, groups by `workerId`, and copies files into the two
subfolders. Workers with >1 unique session are quarantined entirely — a human
has no reason to repeat a 5-minute task for a single payment. Files without a
`workerId` (all Run 1/2 files) pass through to `user-data/` automatically.
A `manifest.json` in `quarantined/` logs every decision for audit. Deleting
both subfolders and re-running regenerates them from scratch.

**Quarantine impact (Run 3)**: 297 files from 77 repeat workers quarantined;
360 files (199 single-session workers + 161 without workerId) passed to
analysis.

After quarantine, `flag_bots()` behavioural detection runs on the remaining
sessions (same criteria as Exp 1: zero angles, fast RTs, low variability),
removing additional single-session bots.

**Combined dataset**: After quarantine and bot removal, Exp 2 contains
**282 sessions** with globally unique user IDs assigned at merge time.

---

## Data Pipeline

```
quarantine_workers()        Exp 2 only: classify raw files into quarantined/
                            (repeat workers) and user-data/ (clean files).
                            Raw directory is never modified.
    ↓
load_from_json()            Parse trials, extract workerId, apply angle
                            sign-correction, derive towards/away, resolve
                            face columns.  Exp 2 loads from user-data/.
    ↓
remove_bad_sessions()       Remove bot-flagged participants (both Exp 1 and
                            Exp 2) based on behavioural criteria
    ↓
validate_trials()           Mark valid = (3° < angle < 40°)
    ↓
balance_cascade()           Single-pass pair invalidation: if one trial in a
                            towards/away matched pair is invalid, its partner
                            is also invalidated
    ↓
select_trials('valid==True')  Final valid trial set for D computation
    ↓
compute_d()                 Per-participant D → one-sample t-test vs 0
```

**Pairing logic for cascade balance**: trials are paired within each
participant by (tube_type_index, tip_direction, condition) — one towards and
one away trial per combination. If the towards trial is invalid (angle out of
range), the matching away trial is also invalidated and vice versa. Only a
single pass is applied; cascading further would propagate invalidation
arbitrarily.

**Angle cutoffs**: `3° < angle < 40°`. The lower bound excludes near-zero
tilts (ambiguous direction or accidental taps). The upper bound excludes
implausibly large tilts. These boundaries were set a priori and validated via
the sensitivity sweep (see below).

---

## Final Sample Sizes (after cleaning and balancing)

### Experiment 1

| Condition     | N (participants with valid D) | D      | SE    | p      |
|---------------|------------------------------|--------|-------|--------|
| Eyes Open     | 577                          | +0.237°| 0.114 | 0.039 * |
| Eyes Covered  | 579                          | +0.011°| 0.110 | 0.921  |

Total valid trials: 15,533 (from 21,449 raw trials across 663 sessions).

### Experiment 2

| Face  | N   | D       | SE    | p     |
|-------|-----|---------|-------|-------|
| ID015 | 275 | −0.102° | 0.209 | 0.627 |
| ID017 | 162 | +0.158° | 0.246 | 0.522 |

ID030 excluded from current analysis (Run 2 data set aside pending more
ID015 + ID017 data). 282 sessions entering the pipeline after quarantine
and bot removal.

---

## Key Result

**Experiment 1** produces a clean double dissociation:
- Eyes Open: D = +0.237°, p = 0.039 — **significant** directional tilt bias
  toward the face.
- Eyes Covered: D = +0.011°, p = 0.921 — **null**, same face, same task,
  blindfolded.

This dissociation rules out motor artefacts, tube geometry effects, and
procedural confounds. The effect is driven specifically by the visible face.

**Experiment 2** is non-significant for both faces after three collection runs
and quarantine of bot-contaminated sessions. Exp 2 now has 162–275 participants
per face — still short of the ~400–500 per face estimated from Exp 1's effect
size. The direction of D for ID017 (+0.158°) remains positive and consistent
with the Exp 1 result. ID015 shows a small negative D (−0.102°), which may
reflect the low-threat face failing to elicit peripersonal engagement.
MTurk bot contamination in Run 3 was severe (77 repeat workers, 297 files
quarantined), reducing the usable yield significantly.

---

## Sensitivity Analysis

To confirm the Exp 1 result is not an artefact of the chosen angle cutoffs,
a full sweep was performed over:
- Min cutoff: 0° to 10° (step 1°)
- Max cutoff: 35° to 55° (step 1°)
— 231 combinations per condition.

At every cutoff pair the full balance-cascade is re-applied and D and p are
recomputed. Results:
- **Eyes Open**: significant (p < 0.05) across a broad contiguous region of
  the cutoff space, centred on the standard cutoff (lo=3°, hi=40°). The
  result is stable, not cutoff-dependent.
- **Eyes Covered**: uniformly null across all 231 combinations. No cutoff
  recovers significance.

The double dissociation holds across the entire cutoff space.

---

## User Angle vs. Tube Critical Angle

A secondary analysis examined whether participants calibrate their tilt to
tube difficulty. Averaging per-participant mean angles across tube types:

| Tube | Critical angle | User mean angle | Slope tracking |
|------|---------------|-----------------|---------------|
| [0] 24×130 | 10.5° | 12.8° | ≈ calibrated |
| [1,2] avg  | 20.3° | 16.0° | undershoot    |
| [3] 48×65  | 36.4° | 19.6° | large undershoot |

Overall r = 0.43 (p < .001) across participants and tube types. Median
per-user slope = 0.31 — participants increase their tilt by only ~0.31° for
every 1° increase in tube difficulty. 89% of users show a positive slope
(correct direction) but most are anchored to a personal comfortable tilt
angle and do not adequately scale to harder tubes. This has no bearing on the
D result (the critical angle is the same for both towards and away trials of
any given tube type) but contextualises participant behaviour in the task.

---

## What Has Not Yet Been Done

- **Power analysis / sample size planning** for additional Exp 2 runs. Current
  N per face (275) is below the estimated ~400–500 needed.
- **Individual differences**: some participants may show reliably larger D;
  this has not been modelled.
- **Threat modulation test**: a direct comparison of D across ID015 vs ID017
  is underpowered but not yet formally tested.
- **ID030 reintegration**: Run 2 data (ID015 + ID030, 150 participants) is
  collected but set aside. Once the ID015/ID017 pairing has sufficient power,
  ID030 can be brought back for a three-face threat-level comparison.
- **Delta analysis**: a sensitivity sweep on `delta = angle − arctan(W/H)`
  (normalising tilt relative to tube difficulty) has been scoped but not yet
  implemented.
- **Replication**: Exp 1 has been run once. A direct replication or
  pre-registered follow-up has not been conducted.

---

## Code and Reproducibility

All analysis is implemented in Python using a pipeline of modular functions:
- `load.py` — raw JSON/CSV ingestion, angle transforms, bot detection,
  worker quarantine (`quarantine_workers()`)
- `sessions.py` — session management, pipeline chaining
- `compute.py` — D computation, sensitivity sweep
- `plots.py` — all figures

The full pipeline is re-run from raw data with a single command:

    python tube_analysis.py

This first runs quarantine (classifying Exp 2 raw files into `quarantined/`
and `user-data/` subfolders), then loads, cleans, and analyses.

Output: `figure_d_bars.png`, `sensitivity_exp1.png`, `sensitivity_exp2.png`

No manual data manipulation steps exist between raw JSON and final results.
The raw data directory is never modified; quarantine output folders can be
deleted and regenerated at any time.
