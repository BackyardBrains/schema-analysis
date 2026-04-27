# Starfield Replication Report: Technical Trajectory and Next-Step Recommendations

Date: 2026-04-24

## Purpose

This report summarizes the Starfield experiment series as a replication attempt of Guterstam and Graziano's 2020 Progress in Neurobiology paper, "Implied motion as a possible mechanism for encoding other people's attention." The goal is to reconstruct what was actually run in each batch, what changed in the code over time, what the accepted participant data show, and what should be changed in the next run if the aim is a clean replication for an MIT Press book discussion.

The central question is whether a static image of an agent attending to an object produces a motion-adaptation-like reaction time effect. In the 2020 paper, the relevant behavioral signature is:

`Delta RT = mean RT(congruent motion) - mean RT(incongruent motion)`

A positive Delta RT means participants are slower when the random-dot motion moves in the same direction as the implied attention/motion. In the original Experiment 2, this effect appeared for a sighted face looking at a tree, but disappeared when the face was blindfolded.

## Data Used

The current accepted data come from `data/star/raw/user-data/`, which is rebuilt from raw files by the quarantine step. Raw files remain in `data/star/raw/`. Quarantined files are copied to `data/star/raw/quarantined/` with a `manifest.json` audit trail.

The current quarantine logic removes:

- repeat MTurk workers
- repeat Prolific workers
- fake Prolific IDs that are not valid 24-character hex IDs
- face/prototype test sessions with no valid platform ID

After quarantine:

- Raw JSON files: 1303
- Quarantined files: 182
- User-data files: 1121
- Unique participants loaded: 1120

For the statistics below, I used the same basic acceptance filters currently implemented in `schema_analysis/star/load.py`:

- trial RT > 200 ms
- participant overall accuracy >= 80%
- RT means computed on correct trials only
- participant-level paired differences computed as congruent minus incongruent
- one-sample t-tests of participant-level Delta RT against 0

The analysis outputs used for this report were also saved as:

- `results/star_batch_summary.csv`
- `results/star_batch_group_stats.csv`
- `results/star_snapshot_parameters.csv`

## Target: Guterstam and Graziano 2020

The 2020 target experiment used a within-trial motion adaptation design. A first stimulus was shown for 1.5 s, followed by a random-dot motion test. Participants made a speeded left/right motion judgment. If the first stimulus was encoded as motion in a direction, the same-direction test motion should be slower.

Technical details from the 2020 paper:

- fixation: 1.5 s
- adaptor image: 1.5 s
- random-dot motion response window: up to 2 s
- aperture: 5 deg x 5 deg
- dot density: 50 dots / deg^2
- dot diameter: 0.05 deg
- dot velocity: 2 deg/s
- dot lifetime: 200 ms
- coherence: 40%
- response: left/right arrow key, as quickly as possible
- Experiment 2 trials: 120 total, 30 per condition
- Experiment 2 conditions:
  - sighted/congruent
  - sighted/incongruent
  - blindfold/congruent
  - blindfold/incongruent

Original Experiment 2 result:

- sighted: Delta RT approximately +22 ms, significant
- blindfold: Delta RT approximately -3 ms, not significant

The key replication target is therefore not simply "any positive congruency effect." It is specifically a positive congruency effect for a sighted attending face, paired with no effect in the blindfold control.

## Critical Implementation Finding: The Interleaved Dot Speed Bug

One major undocumented issue was present through v1.8.6.

The RDK implementation used `interleavedSets = 3`. Only one set of dots was updated on each animation frame. In versions through v1.8.6, dot displacement was:

```js
d.x += d.vx * dtSec;
d.y += d.vy * dtSec;
```

Because each dot set moved only once every three frames, the effective speed was one-third of the declared `dotSpeedDegPerSec`.

This means:

| Version range | Declared speed | Effective speed |
|---|---:|---:|
| v1.0, v1.3 | 2.0 deg/s | 0.67 deg/s |
| v1.4-v1.7 | 2.4 deg/s | 0.80 deg/s |
| v1.8.6 | 3.0 deg/s | 1.00 deg/s |
| v1.9.2-v1.10 | 2.0 deg/s | 2.00 deg/s |

The bug was fixed in v1.9.2 and v1.10 by multiplying displacement by `numSets`:

```js
d.x += d.vx * (dtSec * numSets);
d.y += d.vy * (dtSec * numSets);
```

This is probably one of the most important mechanical explanations for why early runs may not behave like the 2020 paradigm. The nominal parameters looked close, but the actual dot speed was much slower than intended.

## Batch-by-Batch Trajectory

### Batch 1: `rdk-bars` v1.0

Dates: 2025-11-06 to 2025-11-07

Sessions: 68 unique sessions, 37 accepted participants

Design: motion adaptation control with bars/grating-like adaptor rather than face/tree.

Code mechanics:

- 30 trials/session
- 40% coherence
- 5 deg aperture
- 50 dots / deg^2
- dot diameter 0.05 deg
- declared speed 2.0 deg/s, effective speed 0.67 deg/s because of interleaving bug
- RDK aperture border present
- response window 2 s

Result:

| Condition | N | Congruent RT | Incongruent RT | Delta RT | p |
|---|---:|---:|---:|---:|---:|
| bars | 37 | 1121.9 ms | 986.6 ms | +135.3 ms | <0.001 |

Interpretation:

This batch shows that the general system can produce a large congruency/adaptation-like effect, even with the speed bug. It is not a face replication, but it is useful as a sanity check that the task and Delta RT metric can detect a strong low-level adaptation signal.

### Batch 2: `rdk-face-1` v1.3

Dates: 2025-11-13 to 2025-11-15

Sessions: 119, accepted participants: 95

Design: first face/blindfold version close in logic to the 2020 Experiment 2.

Important code-state nuance:

- Early v1.3 sessions had 40 trials/session.
- Later v1.3 sessions had 48 trials/session after a mid-run change to `trialsPerCondition`.
- Accepted participants: 26 in the 40-trial code state and 69 in the 48-trial code state.

Code mechanics:

- sighted and blindfold trials
- congruent and incongruent motion
- 40% coherence
- 50 dots / deg^2
- dot diameter 0.05 deg
- declared speed 2.0 deg/s, effective speed 0.67 deg/s
- RDK aperture border present
- response window 2 s
- `eyesOpen` saved
- face direction not manipulated; unspecified direction should be treated as `towards`

Result:

| Condition | N | Congruent RT | Incongruent RT | Delta RT | p |
|---|---:|---:|---:|---:|---:|
| sighted/towards | 95 | 1057.9 ms | 1059.5 ms | -1.6 ms | 0.876 |
| blindfold/towards | 95 | 1065.4 ms | 1050.5 ms | +14.9 ms | 0.161 |

Interpretation:

This does not replicate the target pattern. The sighted condition has essentially no effect. The blindfold condition trends positive, which is the wrong direction for the original control logic.

Possible technical contributors:

- effective RDK speed was only about one-third of intended
- RDK border was present
- there was a mid-run trial-count change
- online deployment lacked eye tracking and fixed viewing distance

### Batch 3: `rdk-face-1` v1.4

Dates: 2025-12-02 to 2025-12-08

Sessions: 145, accepted participants: 99

Design: sighted/blindfold replication attempt.

Important code-state nuance:

- Started with density 16 dots / deg^2.
- Mid-run on Dec 5, density was changed to 50 dots / deg^2.
- Accepted sessions before Dec 5: 55.
- Accepted sessions Dec 5 or later: 44.

Code mechanics:

- 48 trials/session
- 12 trials per condition
- 40% coherence
- dot diameter 0.05 deg
- declared speed 2.4 deg/s, effective speed 0.80 deg/s
- response window shortened to 1 s
- RDK aperture shifted upward (`centerY = height / 2 - 50`)
- RDK aperture border present
- `eyesOpen` saved

Result:

| Condition | N | Congruent RT | Incongruent RT | Delta RT | p |
|---|---:|---:|---:|---:|---:|
| sighted/towards | 99 | 748.4 ms | 751.0 ms | -2.6 ms | 0.458 |
| blindfold/towards | 99 | 746.0 ms | 749.0 ms | -3.0 ms | 0.380 |

Interpretation:

This run also does not replicate the sighted-face effect. The shortened 1 s response window and mid-run density change make it difficult to interpret. RTs are much faster than v1.3, which may reflect the shorter time window, task pressure, or changed RDK visibility.

### Batch 4: `rdk-face-1` v1.5

Date: 2026-01-27

Sessions: 79, accepted participants: 48

Design: sighted-only Prolific run.

Code mechanics:

- 120 trials/session
- sighted only
- no blindfold control
- 60 congruent and 60 incongruent trials
- coherence lowered to 0.30
- 16 dots / deg^2
- dot diameter 0.05 deg
- declared speed 2.4 deg/s, effective speed 0.80 deg/s
- RDK aperture shifted upward (`centerY = height / 2 - 75`)
- RDK border present
- switched from MTurk-style data capture to Prolific

Result:

| Condition | N | Congruent RT | Incongruent RT | Delta RT | p |
|---|---:|---:|---:|---:|---:|
| sighted/towards | 48 | 1024.9 ms | 1002.2 ms | +22.7 ms | 0.006 |

Interpretation:

This is the strongest face-like positive effect in the sequence and is numerically close to the 2020 Experiment 2 sighted result. But it cannot establish the original claim because the blindfold control was absent. It shows that the system can produce the predicted effect under some conditions, but not that the effect is specific to visual attention.

### Batch 5: `rdk-face-1` v1.6

Dates: 2026-01-30 to 2026-02-03

Sessions: 120, accepted participants: 100

Design: sighted/blindfold Prolific run.

Code mechanics:

- 120 trials/session
- sighted and blindfold
- congruent and incongruent
- coherence returned to 0.40
- 16 dots / deg^2
- dot diameter 0.05 deg
- declared speed 2.4 deg/s, effective speed 0.80 deg/s
- RDK aperture shifted upward (`centerY = height / 2 - 75`)
- RDK border present
- `file_version` stayed at 1.5 despite `experiment_version` 1.6

Result:

| Condition | N | Congruent RT | Incongruent RT | Delta RT | p |
|---|---:|---:|---:|---:|---:|
| sighted/towards | 100 | 964.8 ms | 948.9 ms | +16.0 ms | 0.013 |
| blindfold/towards | 100 | 963.9 ms | 949.5 ms | +14.4 ms | 0.062 |

Interpretation:

This run has a sighted effect, but the blindfold effect is nearly the same size. That is the central problem for a direct replication. It suggests the effect may not have been specific to seeing eyes attending to an object. Possible explanations include low-level directionality, the blindfold stimulus itself, geometric alignment, residual task structure, or the RDK parameters producing a general congruency bias.

### Batch 6: `rdk-face-1` v1.7

Dates: 2026-02-19 to 2026-02-20

Sessions: 117, accepted participants: 87

Design: sighted/blindfold Prolific run.

Important code-state nuance:

The saved v1.7 version-bump snapshot from Feb 10 is not the code state that most accepted participants saw. Two Feb 19 commits occurred before the accepted session wave:

- density changed from 16 to 50 dots / deg^2
- declared speed changed from 2.4 to 2.0 deg/s
- the vertical offset was corrected from `height / 2 - 75` to `height / 2`

However, the interleaving bug was still present, so effective speed was still 0.67 deg/s.

Code mechanics during accepted collection:

- 120 trials/session
- sighted and blindfold
- 50 dots / deg^2
- dot diameter 0.05 deg
- declared speed 2.0 deg/s, effective speed 0.67 deg/s
- RDK centered vertically
- RDK border still present

Result:

| Condition | N | Congruent RT | Incongruent RT | Delta RT | p |
|---|---:|---:|---:|---:|---:|
| sighted/towards | 87 | 967.5 ms | 959.1 ms | +8.3 ms | 0.298 |
| blindfold/towards | 87 | 975.0 ms | 963.3 ms | +11.7 ms | 0.150 |

Interpretation:

Correcting the vertical alignment and increasing density did not produce a clean sighted-specific replication. Both conditions trend positive, and neither is significant.

This is relevant to the concern about face position. The eye/RDK alignment change may have been good experimental hygiene, but by itself it does not appear to explain the earlier blindfold problem. The bigger unresolved mechanical issue in v1.7 remained the one-third effective speed.

### Batch 7: `rdk-face-1` v1.8.6

Date: 2026-03-06

Sessions: 121, accepted participants: 99

Design: shifted from blindfold control to face-towards vs face-away control.

Code mechanics:

- 120 normal RDK trials/session
- 30 trials per condition
- face direction saved as `towards` or `away`
- eyes condition not saved and should be treated as `unknown`
- catch trials added but not included in the RDK trial payload analyzed here
- density 40 dots / deg^2
- dot diameter 0.035 deg
- declared speed 3.0 deg/s, effective speed 1.00 deg/s
- RDK border removed
- RDK centered vertically
- Bob story/narrative present

Result:

| Condition | N | Congruent RT | Incongruent RT | Delta RT | p |
|---|---:|---:|---:|---:|---:|
| unknown eyes, face towards | 99 | 907.0 ms | 897.2 ms | +9.8 ms | 0.122 |
| unknown eyes, face away | 99 | 887.8 ms | 918.9 ms | -31.1 ms | <0.001 |

Interpretation:

The towards condition trends in the predicted positive direction, but the away condition shows a strong negative effect. This is not the original blindfold control, but it is theoretically interesting: it suggests the face-away manipulation changed behavior strongly, perhaps more like a directional compatibility or spatial cueing effect than a simple absence of attention-beam adaptation.

Because eyes condition was not saved, this run cannot answer the sighted vs blindfold question.

### Batch 8: `rdk-face-1` v1.9.2

Dates: 2026-03-29 to 2026-03-30

Sessions: 117, accepted participants: 94

Design: face-towards vs face-away run.

Known logging problem:

The run did include sighted/blindfold variation in the stimulus, but the relevant trial condition was not saved. Therefore, all trials must be treated as `eyes_condition = unknown` for analysis. The run can still contribute to pooled face-direction analyses, but not to a sighted-vs-blindfold test.

Code mechanics:

- 120 trials/session
- face direction saved
- eyes condition not saved
- density 50 dots / deg^2
- dot diameter 0.05 deg
- declared speed 2.0 deg/s
- interleaving speed bug fixed, so effective speed 2.0 deg/s
- RDK border removed
- RDK centered vertically

Result:

| Condition | N | Congruent RT | Incongruent RT | Delta RT | p |
|---|---:|---:|---:|---:|---:|
| unknown eyes, face towards | 94 | 881.4 ms | 866.9 ms | +14.5 ms | 0.059 |
| unknown eyes, face away | 94 | 857.7 ms | 897.5 ms | -39.8 ms | <0.001 |

Interpretation:

This is the first run with the corrected RDK speed. The towards condition is close to a positive effect, and the away condition again strongly reverses. This supports the idea that the face-direction manipulation matters, but it still cannot support a direct Guterstam 2020 blindfold replication because the sighted/blindfold state was not saved.

### Batch 9: `rdk-face-1` v1.10

Dates: 2026-04-14 to 2026-04-16

Sessions: 234, accepted participants: 204

Design: returned to sighted/blindfold, with catch trials and centered RDK.

Important code-state nuance:

There were two accepted code states within v1.10:

- 103 accepted participants from Apr 14-15 had no trial-level condition saved.
- 101 accepted participants from Apr 16 had trial-level `condition` saved as `normalFace` or `blindfold`.

Thus v1.10 must be analyzed as:

- unknown eyes/towards
- sighted/towards
- blindfold/towards

Code mechanics:

- 70 trials/session, not 120
- face direction was towards only
- density 50 dots / deg^2
- dot diameter 0.05 deg
- declared speed 2.0 deg/s
- interleaving speed bug fixed, effective speed 2.0 deg/s
- RDK border removed
- RDK centered vertically
- condition logging fixed only for the later accepted sessions

Result:

| Condition | N | Congruent RT | Incongruent RT | Delta RT | p |
|---|---:|---:|---:|---:|---:|
| unknown eyes, towards | 103 | 849.0 ms | 843.8 ms | +5.2 ms | 0.385 |
| sighted/towards | 101 | 820.3 ms | 821.1 ms | -0.9 ms | 0.915 |
| blindfold/towards | 101 | 822.9 ms | 816.6 ms | +6.3 ms | 0.377 |

Interpretation:

The corrected condition-logging subset does not replicate the 2020 sighted effect. It also does not show a strong blindfold effect. In one sense this is cleaner than v1.6, because the blindfold no longer falsely reproduces the sighted effect. But the target sighted effect is also absent.

The major interpretive concern is statistical/design power at the condition level. The run has only 70 trials/session, split across sighted/blindfold and congruent/incongruent, rather than the 120 trials and 30 per condition used in the 2020 Experiment 2. This may reduce within-participant stability.

## What Could Prevent Replication?

### 1. The RDK speed bug

This is the clearest mechanical problem. For v1.0-v1.8.6, the RDK speed was effectively one-third of the declared speed because of the interleaved dot-set implementation. Early settings may have looked close to the 2020 paper but were not actually close.

### 2. Multiple mid-run code changes under the same version number

The data include several cases where a single `experiment_version` hides multiple deployed code states:

- v1.3: trial count changed from 40 to 48
- v1.4: density changed from 16 to 50
- v1.7: density, speed, and vertical RDK alignment changed before the accepted session wave
- v1.10: condition logging changed during the run

For a book-quality replication narrative, each run should have a frozen commit and no mid-run edits.

### 3. Condition logging failures

The v1.8.6 and v1.9.2 runs cannot answer the blindfold/sighted question because the eyes condition was not saved. v1.10 was only partially fixed: about half of accepted participants had condition saved, while earlier v1.10 participants must be treated as `unknown`.

### 4. Trial counts changed substantially

The original Experiment 2 used 120 trials/session, 30 per condition. Our runs varied:

- v1.3: 40 or 48 trials
- v1.4: 48 trials
- v1.5-v1.9.2: 120 trials
- v1.10: 70 trials

The strongest direct sighted-only result was v1.5 with 120 trials, but it lacked the blindfold control. The corrected v1.10 logging run had only 70 trials.

### 5. RDK visibility and task difficulty changed repeatedly

Density, speed, coherence, dot size, border, and response duration all changed. These are not cosmetic. They directly affect RT, accuracy, and motion sensitivity.

Especially important changes:

- density 16 vs 40 vs 50 dots / deg^2
- effective speed 0.67 vs 0.80 vs 1.00 vs 2.00 deg/s
- coherence 0.30 vs 0.40
- response window 1 s vs 2 s
- border present vs absent
- random-noise dots initially teleported rather than moving with random velocities

### 6. Face/RDK geometry changed

The RDK aperture was initially vertically centered, then shifted upward, then centered again. The stated rationale was to align the RDK with the face/tree axis or eye plane. This is worth documenting, but the data do not show that geometry alone solved the problem. v1.7 had improved centering and denser dots, yet still did not produce a sighted-specific effect.

### 7. Narrative context may matter

The original 2020 Experiment 2 did not use a named character story. It told participants the face/tree image was irrelevant to the dot task. Some of our later versions used "Bob" story framing and green catch events. This may increase explicit social interpretation or attentional set. That is not necessarily bad for a demonstration, but it moves the task away from a strict replication.

## Current Best Interpretation

The sequence tells a coherent but complicated story:

1. The task can detect motion adaptation strongly in a non-face control (`rdk-bars`).
2. Early face runs did not cleanly replicate the sighted-specific 2020 pattern.
3. A sighted-only run (v1.5) produced a positive Delta RT close to the original magnitude, but without a blindfold control.
4. When blindfold was reintroduced (v1.6), the blindfold condition also produced a positive effect, undermining the original interpretation.
5. Centering and density changes (v1.7) did not solve this.
6. The face-towards vs face-away runs (v1.8.6/v1.9.2) produced a meaningful directional pattern, especially a strong negative away effect, but cannot answer the blindfold question because eyes condition was not saved.
7. The corrected v1.10 subset did not show the target sighted effect.

So the project has not yet produced a clean direct replication of Guterstam and Graziano 2020 Experiment 2. It has produced useful evidence about which mechanical choices matter and which design variants are interpretable.

## Recommended Next Run

If the goal is a book-quality replication of the 2020 Progress in Neurobiology face result, the next run should be deliberately boring: one frozen code version, one clean design, one primary analysis.

### Recommended design

Replicate 2020 Experiment 2 as directly as possible:

- 120 trials/session
- 30 trials per condition
- conditions:
  - sighted/towards/congruent
  - sighted/towards/incongruent
  - blindfold/towards/congruent
  - blindfold/towards/incongruent
- do not include face-away in this run
- do not include catch trials in the main timing stream unless they are clearly separated and excluded
- use a simple instruction: the face/tree image is irrelevant to the dot task
- avoid the "Bob" narrative for the strict replication run

### Recommended RDK settings

Use the 2020 parameters:

- aperture: 5 deg x 5 deg
- density: 50 dots / deg^2
- dot diameter: 0.05 deg
- effective speed: 2 deg/s
- lifetime: 200 ms
- coherence: 40%
- adaptor/face display: 1.5 s
- response window: 2 s
- no aperture border unless there is a specific reason to include it

Implementation requirement:

- Either update every dot on every animation frame, or keep the interleaved sets but multiply displacement by `numSets`.
- Store both declared and effective speed in the output.
- Store observed frame rate per trial.

### Required data fields

Each RDK trial should save:

- `experiment_version`
- `file_version`
- `commit_sha`
- `prolificPid`
- `uuid`
- `trial_index`
- `eyes_condition`: `sighted` or `blindfold`
- `face_direction`: `towards`
- `congruent`: true/false
- `faceOnLeft`
- `impliedDirection`
- `testDirection`
- `response`
- `rt`
- `correct`
- `coherence`
- `apertureDeg`
- `densityDotsPerDeg2`
- `dotDiameterDeg`
- `dotSpeedDegPerSec`
- `effectiveDotSpeedDegPerSec`
- `lifetimeMs`
- `avgFps`
- `minFps`
- `maxFps`

### Recommended exclusion rules

Match the spirit of the 2020 paper:

- practice gate before the main experiment
- 10 practice trials
- repeat practice up to 4 times
- require >=80% practice accuracy
- exclude participants with poor main-task accuracy
- exclude implausibly fast responders
- quarantine repeat platform IDs and fake IDs before analysis

Because this is online rather than lab-based, also record and report:

- browser
- OS
- viewport size
- screen resolution
- measured pixels-per-degree or calibration proxy if available
- measured frame rate

### Recommended reporting strategy

For the book narrative, report:

1. The original 2020 effect and why it matters.
2. The fact that our first iterations did not simply "fail"; they uncovered important implementation sensitivities.
3. The RDK interleaving bug as a concrete technical lesson.
4. The blindfold control as the decisive interpretive test.
5. The next clean run as the actual replication attempt.

## Bottom Line

The most likely technical blockers to replication so far were not the face position alone. The larger issues were:

- the one-third effective dot speed bug
- unstable code during active data collection
- condition logging failures
- changing trial counts and RDK parameters across runs
- mixing direct blindfold replication with later face-away control logic

The next run should freeze the code, use corrected RDK mechanics, return to the original 120-trial sighted/blindfold design, and store every condition explicitly. That would give the cleanest possible test of whether we can reproduce the 2020 sighted-face Delta RT effect while preserving the blindfold null control.
