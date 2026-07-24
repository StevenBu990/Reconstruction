# Fragment Repair Pipeline

Automated pipeline for repairing fragmented 3D objects (Breaking Bad dataset) using a vision-language model (Gemini). Given a photo of a damaged object, the pipeline reconstructs what the complete object should look like, then scores the reconstruction against ground truth.

## Pipeline overview

```
Blender → damage object → render → Gemini reconstruction → SIFT coregistration → IoU/Dice eval
```

1. **Break it** — hide certain 3D fragments to simulate damage (e.g. "neck broken, mild or severe")
2. **Render it** — Blender renders the damaged object (`input.png`) and the complete object (`ground_truth.png`) from the same camera
3. **Reconstruct it** — Gemini is shown `input.png` and asked to generate `generated.png`, its guess at the complete object
4. **Align it** — SIFT feature-matching warps `generated.png` onto `ground_truth.png`'s coordinate frame (falls back to manual point-click alignment when SIFT fails)
5. **Score it** — IoU and Dice measure how well the reconstructed missing region overlaps the true missing region

15 objects total: 10 fully automated (beer_bottle, wine_glass, vase, wine_bottle, bottle, spoon, ring, drink_bottle, drinking_utensil, pill_bottle), 5 using an earlier manual pipeline (ramekin_cup, teapot, mug, plate, bowl).

## Repo structure

```
src/
├── pipeline/        Blender rendering, rotation/adjacency setup, experiment orchestration
├── reconstruction/  Gemini reconstruction calls, marker drawing, AI view judge variants
├── evaluation/      Coregistration, mask extraction, IoU/Dice/centroid scoring, diagnostics
└── legacy/          Superseded scripts, kept for reference only

configs/             Fragment maps, view presets, experiment manifests
project_data/        objects.json, damage.json — object/damage descriptions used in prompts
docs/                Full command reference, detailed running notes, meeting summaries
deliverables/        Compiled results (spreadsheets, reports) for meetings
```

## Setup

Two conda environments are required:

```bash
conda activate Reconstruction   # for Gemini calls (FullMethod.py, ai_view_judge.py)
# or
conda activate Evaluation       # for cv2/matplotlib scripts (coregistration, masking, scoring)

echo $GEMINI_API_KEY             # must print your real key, not empty
```

## Quick start

Full command reference with every script's exact arguments lives in **`docs/COMMANDS_REFERENCE.md`**. The short version:

```bash
# Run the full batch pipeline (resumable)
python src/pipeline/RunExperiment.py --manifest configs/experiment_manifest.json ...

# Coregister + evaluate a batch
python src/evaluation/CoregisterAndEvaluate.py --manifest configs/experiment_manifest.json ...

# Aggregate results into one CSV
python src/evaluation/aggregate_metrics.py --manifest configs/experiment_manifest.json --output results/aggregated_metrics.csv
```

## Experimental tracks

Two techniques have been tested for improving reconstruction quality beyond the fixed-view baseline:

- **Shape-drawing marker** (`draw_marker.py`) — draws a smooth outline on the damaged input showing exactly where the missing piece belongs, before asking Gemini to reconstruct. **6/6 clean improvements** across every object tested so far.
- **AI View Judge** (`ai_view_judge.py`) — renders 5 candidate camera angles (0-180°, 45° increments) and has Gemini pick which view gives the best chance of a successful repair. Mixed results — one strong win, but a positional-bias question (does it just default to the first image shown?) remains unresolved.
- **Human-guided verbal-description judge** (`ai_view_judge_verbal.py`) — a human describes what makes a good view; tested once, found unreliable (see `docs/PROJECT_STATE.md` for details).

Full results, per-object statistics (mean/min/max/std/error), and known limitations are compiled in **`deliverables/marker_and_judge_results.xlsx`**.

## Known limitations

- **Mask-extraction "edge halo"**: residual sub-pixel alignment noise can trace a false line along a long object silhouette edge, inflating `area_ratio`/`centroid_distance` (but not IoU/Dice, which are cropped and remain reliable). A diagnostic/cleanup tool (`mask_halo_diagnostic.py`) fixes this on some objects, not others — see `docs/PROJECT_STATE.md` Known Issue #4 for the full breakdown.
- **SIFT alignment** fails on low-texture geometry (confirmed on spoon and plate). Falls back to `manual_align_matplotlib.py`.
- **Reconstruction has real run-to-run variance** — repeated Gemini calls on identical input can differ by ~0.09 IoU. Single before/after comparisons should be read with this in mind.

## Detailed notes

`docs/PROJECT_STATE.md` is the full working log — session history, every known issue with root causes, and an "Immediate next steps" section kept up to date for picking work back up.
