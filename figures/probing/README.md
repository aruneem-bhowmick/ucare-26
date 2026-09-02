# Probing pilot artifacts

Real (non-mocked) extract → probe → plot results from the Track-A pipeline
validation pass. These are git-tracked, unlike `outputs/` (gitignored,
regenerable from cache). This directory is where result figures worth
keeping around and citing live.

## Pilot v1: single seed, Pythia-160M (2026-09-01)

All six Track-A tasks, one seed (42), Pythia-160M, CPU. First real
cross-family look at the project's central hypothesis (H1: factual/semantic
tasks become linearly decodable earlier than algorithmic ones). Small N and
a single seed, so this is a pipeline-correctness and first-signal check,
not yet a statistically supported H1 test.

| Task | Family | N | Best layer (/12) | Score | Chance | Selectivity |
|---|---|---|---|---|---|---|
| [SST-2](sst2_score_vs_layer.png) | shallow semantic | 64 | 5 | 0.673 | 0.50 | 0.208 |
| [MRPC](mrpc_score_vs_layer.png) | shallow semantic | 90 | 7 | 0.600 | 0.68 | 0.144 |
| [LAMA/T-REx](lama_trex_score_vs_layer.png) | factual lookup | 90 | 10 | 0.911 | 0.33 | 0.489 |
| [Periodic table](periodic_table_score_vs_layer.png) | factual lookup (R², continuous) | 400 | 1 | 0.634 | 0.00 | 1.003 |
| [Dyck-3](dyck_score_vs_layer.png) | algorithmic | 300 | 10 | 0.673 | 0.50 | 0.130 |
| [Modular arithmetic](modular_arithmetic_score_vs_layer.png) | algorithmic | 700 | 6 | 0.863 | 0.14 | 0.720 |

**[Combined cross-task comparison](cross_task_score_vs_layer.png)**: all six
score-vs-layer curves overlaid, colored by family (blue = factual, green =
shallow semantic, red = algorithmic; solid/dashed distinguishes the two
tasks within a family).

Prediction-depth distributions: [SST-2](sst2_prediction_depth_distribution.png),
[LAMA/T-REx](lama_trex_prediction_depth_distribution.png) (only computed for
the two tasks validated through `src.probing.smoke_test`; the other four ran
through `ExtractionPipeline`/`ProbingPipeline` directly).

Selectivity is well above zero everywhere, so none of this is probe
memorization. Two things worth flagging for later analysis:

- **Modular arithmetic** shows the cleanest H1-consistent signal: near-chance
  at the embedding layer, then a sharp jump to ~0.85 by layer 2, then a
  plateau. It needs a couple of computation steps, then is linearly done.
- **Dyck-3** has the lowest selectivity (0.130) of the six despite a
  respectable raw score, and its score is already high at layer 0. This
  suggests a possible surface-statistic confound (bracket-count/parity cues
  in raw token embeddings), rather than genuine structural decoding. Worth a
  closer look before trusting Dyck as a clean algorithmic probe.

Underlying data: [`data/`](data/), containing per-plot companion CSVs
(`*_score_vs_layer.csv`, `*_prediction_depth_distribution.csv`) and the full
per-layer metrics tables (`pythia-160m_<task>.csv`: score, selectivity,
margin, ECE, prediction depth).

## Pilot v2: 5 seeds × {70M, 160M} (2026-09-01)

Larger N per task (300-700), 5 seeds, both Pythia-70M and 160M, CPU. Real
error bars across sampling seeds (shaded bands below) rather than only
within-fold CV std. First look at whether H1 (factual/semantic tasks become
linearly decodable earlier than algorithmic ones) also holds up as model
scale changes.

| Task | Model | Best layer (of N) | Score | Selectivity |
|---|---|---|---|---|
| [LAMA/T-REx](pilot/lama_trex_score_vs_layer.png) | Pythia-70M | 5/6 | 0.845 | 0.494 |
| LAMA/T-REx | Pythia-160M | 12/12 | 0.867 | 0.494 |
| [Periodic table](pilot/periodic_table_score_vs_layer.png) | Pythia-70M | 1/6 | 0.606 | 0.895 |
| Periodic table | Pythia-160M | 1/12 | 0.634 | 0.920 |
| [SST-2](pilot/sst2_score_vs_layer.png) | Pythia-70M | 4/6 | 0.716 | 0.179 |
| SST-2 | Pythia-160M | 8/12 | 0.742 | 0.235 |
| [MRPC](pilot/mrpc_score_vs_layer.png) | Pythia-70M | 5/6 | 0.644 | 0.068 |
| MRPC | Pythia-160M | 7/12 | 0.697 | 0.124 |
| [Dyck-3](pilot/dyck_score_vs_layer.png) | Pythia-70M | 4/6 | 0.679 | 0.156 |
| Dyck-3 | Pythia-160M | 10/12 | 0.685 | 0.176 |
| [Modular arithmetic](pilot/modular_arithmetic_score_vs_layer.png) | Pythia-70M | 6/6 | 0.980 | 0.829 |
| Modular arithmetic | Pythia-160M | 8/12 | 0.981 | 0.836 |

**[Combined cross-task comparison](pilot/cross_task_score_vs_layer.png)**:
both models side by side, same family-color/task-linestyle scheme as v1, so
the shape of each task's layer-depth curve can be compared across scale
directly. Per-task selectivity-vs-layer companions live alongside the score
plots (`pilot/<task>_selectivity_vs_layer.png`); aggregated
mean/std-across-seeds data for every plotted point is in
[`pilot/data/`](pilot/data/).

Findings from this pass:

- **Modular arithmetic gives the cleanest scale story**: both models start
  near chance at the embedding layer and both plateau at ~0.98, but Pythia-160M
  gets there by ~25% depth while 70M needs ~50-65% depth. Bigger model, same
  ceiling, reached in fewer relative layers — a scale effect on top of the v1
  H1 signal, not just a bigger number.
- **Periodic table stays noisy and non-monotonic in both models** (R² swings
  up and down across layers rather than plateauing, and *declines* at the
  final layer for both 70M and 160M). This tracks with a numerical finding
  from this run: fitting Ridge (this task is regression, `label_type:
  "regression"`) at the lightest regularization_grid value (alpha=0.001) on
  standardized, highly collinear hidden-state features throws scipy
  `LinAlgWarning: Ill-conditioned matrix` — 1,151 times total across this
  run, **100% of them during periodic_table fits** (749 for Pythia-160M, 402
  for Pythia-70M; zero for every other task, since only periodic_table uses
  the closed-form Ridge solve — everything else is classification via
  `LogisticRegression`/lbfgs, which doesn't hit this code path). Benign
  (cross-validation still picks the best-scoring alpha off the sweep,
  and sklearn returns a least-squares solution regardless), but it's a real
  signal that this task's representations are hard to fit stably at low
  regularization, consistent with the noisy curve. Worth revisiting with a
  narrower/higher-alpha regularization_grid for this task specifically.
- **MRPC selectivity is very low** (0.068 for 70M, 0.124 for 160M) — even
  lower than v1's flagged Dyck-3 (0.130). Worth the same scrutiny v1 gave
  Dyck: check whether the probe is reading paraphrase structure or a
  surface-statistic shortcut (e.g. lexical overlap between the sentence
  pair), since selectivity this low is close to memorization territory.
- Everywhere else, selectivity comfortably clears the v1 tasks' range, so
  the rest of the signal looks like genuine structure rather than probe
  memorization.

### Relation to H1

H1 (`spec/proposal/PROPOSAL.md:43`): *"Factual recall and shallow semantic
tasks achieve high linear probe accuracy at significantly earlier layers than
algorithmic reasoning tasks."* Two figures visualize this directly:

- **[Family small multiples](pilot/h1_family_small_multiples.png)** — all
  six score-vs-layer curves, faceted into a 3x2 grid and ordered top-to-bottom
  the way H1 predicts (factual recall, shallow semantic, algorithmic), so a
  top-to-bottom read shows whether the plateau moves later as predicted.
- **[Depth-to-ceiling summary](pilot/h1_depth_to_ceiling_summary.png)** —
  each task/model curve collapsed to the layer-depth-fraction at which it
  first reaches 90% of its own max score, grouped by family, with a family-mean
  band per model. **This is a proxy, not the project's `prediction_depth`
  metric** (`src/metrics/prediction_depth.py`, populated by
  `ProbingPipeline.run()` as `prediction_depth_mean`) — `run_pilot.py` only
  retained `score`/`selectivity` in `pilot_results.csv`, so a rigorous
  per-example depth test isn't available from this run. Underlying data:
  [`pilot/data/h1_depth_to_ceiling_summary.csv`](pilot/data/h1_depth_to_ceiling_summary.csv).

Reading both: **support for H1 is mixed at this pilot's scale**, and
within-family spread is often as large as between-family spread.

- Factual recall's mean sits earliest for both models (0.125-0.167), the one
  clean H1-consistent family result.
- Shallow semantic sits in between (0.208-0.250), as H1 predicts, but only
  barely ahead of algorithmic reasoning on Pythia-160M (0.208 vs. 0.125 — the
  smaller gap, and algorithmic is earlier here, the opposite of H1).
- Algorithmic reasoning's Pythia-70M mean is latest (0.333) — H1-consistent —
  but almost entirely driven by `modular_arithmetic` (0.5); `dyck` alone sits
  as early as anything in the factual/semantic families (0.083-0.167), tracking
  the surface-statistic confound already flagged above rather than genuine
  early algorithmic decodability.
- `lama_trex` (factual recall) never plateaus — it's still climbing at the
  final layer in both models — which is the single data point most in tension
  with H1's "early" claim for factual recall.

Net: `modular_arithmetic` vs. `sst2`/`mrpc` is the cleanest piece of evidence
for H1 in this pilot; `dyck`'s early-high score and `lama_trex`'s late climb
are the two clearest complications. Treat this pilot as validating the
pipeline and surfacing confounds to control for, not as a completed test of
H1 — that needs `prediction_depth_mean` retained and, per
`spec/plans/WEEK2-PLAN.md:137`, is Week 3 scope.
