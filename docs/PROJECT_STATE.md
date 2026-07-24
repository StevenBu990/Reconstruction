# Fragment Repair Pipeline — Project State (as of this session)

## Project goal
Automate a research pipeline for repairing fragmented 3D objects (Breaking Bad dataset) using a VLM (Gemini). Original manual workflow: Blender → damage object → render → Gemini reconstruction → SIFT coregistration → IoU/Dice eval. This session built full automation for 10 of 15 objects, then moved into exploratory experiments per meeting notes (AI view judge, red-circle/shape marker guidance).

## Directory layout (all on user's Desktop, macOS, zsh)
- Scripts live directly in `~/Desktop/`
- `~/Desktop/configs/` — fragment_maps/, view_presets.json, candidate_views.json, object_registry.json, experiment_manifest.json
- `~/Desktop/outputs/` — main 160-experiment results (10 objects x 4 damage configs x 4 views)
- `~/Desktop/outputs_judge_test/` — AI-view-judge experiment output (separate from main outputs)
- `~/Desktop/Breaking-Bad-Dataset.github.io/everyday/` — raw dataset
- `~/Desktop/ImageCoregistration/` — user's own SIFT alignment tool (image_register.py + utils.py), NOT modified this session
- `~/Desktop/Teapot1Fratture15/` — one of 5 pre-existing MANUAL objects (handle_simple/severe, spout_simple/severe, each with view1-4 subfolders)
- User's conda envs: Reconstruction (has google-genai) and Evaluation (has cv2/sklearn/matplotlib) - must activate the right one or imports fail

## Scripts built this session (all on Desktop)

Core Blender/data pipeline:
- blender_render.py - headless Blender script. Imports piece_N.obj fragments, applies damage config (hides pieces), applies base_rotation_euler_deg (PCA-computed) + view rig rotation, renders input.png + ground_truth.png + metadata.json per view. Default engine WORKBENCH (matches user's original manual look). --distance_factor controls framing, --light_energy_scale controls brightness. IMPORTANT: forces forward_axis='Y', up_axis='Z' on import to disable Blender's default OBJ axis-conversion (was causing objects to render sideways).
- compute_base_rotation.py - PCA-based rotation computation (no Blender needed). Finds object's longest axis, aligns to Z. Has --flip (aligns to -Z instead, for when PCA picks the wrong end) and --include_pieces (restrict PCA to specific pieces, for asymmetric objects like a kettle where handle/spout skew the whole-object axis).
- check_fragment_adjacency.py - validates fragment "damage configs" geometrically: --remove_set checks a candidate hide-list is internally connected AND doesn't orphan the remainder; --seed X --sizes 1 2 3 auto-searches all valid combos containing piece X.
- resolve_fracture_instances.py / manifest_generator.py - pick best-fit fracture instance per object (closest to target piece count), build final experiment_manifest.json.
- RunExperiment.py - orchestrator: Blender render (once per object) + FullMethod.py (once per experiment). Resumable (skips existing outputs).
- align_helper.py - interactive Blender GUI helper (had context/focus issues, largely superseded by compute_base_rotation.py).

Reconstruction:
- FullMethod.py (user's original, lightly extended) - calls Gemini. Added --metadata/--objects/--damages/--output args (was hardcoded filenames). Added --marked_image (use an edited image instead of the raw damaged input) and --use_marker_note --marker_hint_type {location,shape} (adds a prompt sentence when testing marker experiments).

Coregistration/eval (wraps user's existing image_register.py/utils.py - NEVER modified those):
- CoregisterAndEvaluate.py - orchestrates coregistration (target=ground_truth.png, source=generated.png, since these share the same camera by construction) + eval for the whole batch. Sets up data/result/ subfolders per experiment so image_register.py's hardcoded relative paths work with zero code changes to it.
- extract_missing_mask.py - parameterized version of user's mask-diff logic (background-distance + Otsu threshold). Produces GT_Mask.png (ground_truth vs input) or GT_Fragment.png (aligned reconstruction vs input). Has --kernel_size (morphological cleanup) and --filter_compactness --min_circularity X (removes ring/halo-shaped false positives from imperfect alignment - see KNOWN ISSUES below).
- evaluate.py - parameterized IoU/Dice computation, crops to GT bounding box + padding. --output_name lets you save multiple variants (metrics.json, metrics_marked.json, etc.) without overwriting. Handles empty-GT-mask case gracefully (writes flag="empty_gt_mask" instead of crashing).
- centroid_distance.py - NEW metric complementing IoU: computes centroid delta_x/delta_y and area_ratio between GT_Mask and GT_Fragment. Built because IoU alone can't distinguish "close but wrong shape" from "completely wrong location" - and in one real case (beer_bottle outline test) IoU actively favored a worse, oversized, wrong-shaped reconstruction.
- reset_evaluation.py - deletes coregistration/eval outputs (not renders/reconstructions) for re-testing. --object_ids, --exclude_objects, or --experiments obj/config/view (exact list).
- aggregate_metrics.py - pulls all metrics.json into one CSV + console table, with flag breakdown (empty_gt_mask / bad_alignment / empty_prediction).
- manual_align_matplotlib.py - FALLBACK for when image_register.py's cv2-window manual click mode is unresponsive (known macOS/conda issue). Uses matplotlib ginput for click collection, calls a LOCAL COPY of utils.py's warp() math (minus the cv2.imshow calls that also hang) - same output, working UI.

Meeting-notes experiments (in progress):
- draw_marker.py - draws a marker on input.png at the missing-fragment location (from GT_Mask.png). --shape circle/box (location only) or --shape outline (raw jagged contour - DON'T USE, see below) or --shape smooth_outline (convex hull + polygon simplification - this is the good one).
- ai_view_judge.py - sends N candidate damaged-view images (input.png only, NEVER ground_truth) to Gemini, asks it to pick the best view for reconstruction. Parses a single digit response.
- configs/candidate_views.json - 5 candidate views, 0 to 180 degrees in 45-degree increments, per meeting notes.

## Key established conventions
- Object IDs (10 automated): beer_bottle, wine_glass, vase, wine_bottle, bottle, spoon, ring, drink_bottle, drinking_utensil, pill_bottle
- 5 pre-existing MANUAL objects (different pipeline, not touched by automation): ramekin_cup, teapot, mug, plate, bowl
- Damage config naming: always {part}_{simple|severe} (e.g. "neck_simple") - this is also the exact suffix used in damage.json keys ({object_id}_{part}_{level})
- Fragment maps live at configs/fragment_maps/{object_id}.json: {"base_rotation_euler_deg": [x,y,z], "damage_configs": {"part_simple": ["piece_N"], ...}}
- objects.json / damage.json (Desktop) have entries for all 15 objects now
- Coregistration convention: target=ground_truth.png, source=generated.png (or marked variant) - NOT input.png as target (input and ground_truth share the same camera by construction, so aligning to ground_truth also aligns to input)

## KNOWN ISSUES / LIMITATIONS discovered this session
1. SIFT fails on smooth/low-texture geometry (confirmed on spoon: literally 0-2 matched keypoints in some views). Fallback: manual_align_matplotlib.py. Doesn't scale to 50 objects.
2. Raw jagged mask outlines can be visually misread by the VLM (a jagged trace on a beer bottle neck was reconstructed as something looking like a cap-ish shape - smooth_outline fixed this).
3. IoU alone can be actively misleading - confirmed case where a worse (oversized, wrong-shape) reconstruction scored HIGHER IoU than a better one. Always pair with centroid_distance.py, and always visually check images before trusting numbers.
4. Whole-silhouette mask-difference method breaks down on complex/curved silhouettes (confirmed on teapot): even a good, visually-verified alignment produces a thin/moderate-width "halo" tracing the ENTIRE object outline (not just the true missing region), because residual sub-pixel alignment noise along a long curved boundary is enough to trigger the per-pixel diff everywhere. Tried: larger morphological kernel (didn't work), compactness/circularity filtering (also didn't cleanly separate - the halo and real fragment appear to be ONE connected contour, not separable by shape either). Concluded this is a genuine limitation for curved-silhouette objects, not something parameter tuning fixes. NOT YET RESOLVED - a real open problem.
5. Both bad-alignment (#1) and bad mask-extraction (#4) can look SIMILAR in symptom (crazy centroid distance/area ratio) but have different root causes - always check the alignment images (source_new.png vs target_new.jpg) first, then the extracted GT_Fragment.png mask shape, before diagnosing.

## Results so far
- Full automated batch (10 objects x 160 experiments): mean IoU 0.489, Dice 0.595 (in results/aggregated_metrics.csv on Desktop)
- Combined with 5 manual objects: results/combined_all_15_objects.csv, mean IoU 0.527
- Per-object summary with min/max/std dev: object_summary_with_spread.csv
- Meeting slide deck built: fragment_repair_update.pptx (10 slides)
- Marker experiment (smooth_outline vs baseline), tested tonight:
  - beer_bottle/neck_simple/view_0 (front): baseline IoU=0.396 -> smooth_outline IoU=0.473 (improvement)
  - wine_bottle/neck_simple/view_0 (front): baseline IoU=0.624 -> smooth_outline IoU=0.671 (improvement)
  - teapot handle_simple/view1: BLOCKED by limitation #4 above, no valid comparison obtained
- AI View Judge: tested once on beer_bottle/neck_simple - judge picked view_2 (90 degrees) out of 5 candidates (0/45/90/135/180). Was mid-way through getting IoU/Dice for this pick vs the fixed-front baseline (0.396/0.567) when this document was written - NEXT STEP is to finish that comparison.

## Immediate next steps (pick up here)
1. Finish the beer_bottle AI-view-judge test: run coregistration + eval on the already-generated outputs_judge_test/beer_bottle/neck_simple/view_2/generated.png, compare its IoU/Dice against the front-view baseline (0.396/0.567)
2. Run the AI view judge on 1-2 more object/damage-config combos to get more than n=1
3. Consider testing smooth_outline marker on one more object with reliable coregistration (avoid curved-silhouette objects like teapot until issue #4 is resolved)
4. Decide whether to invest time in solving limitation #4 (curved-silhouette eval) - this affects any future experiments on non-cylindrical objects (vase, ring, drinking_utensil, teapot, mug, bowl, plate)
5. Eventually: build the results table meeting notes ask for - same objects, add columns for judge-selected-view results and marker/drawing results, alongside existing fixed-view baseline
6. Also from notes: "For IoU, show highest IoU, lowest IoU, standard deviation, error bar" - partially done (object_summary_with_spread.csv has min/max/std), could still add visual error bars to a chart
