# Command Reference - Fragment Repair Pipeline
Keep this file. All commands assume you're in ~/Desktop unless a cd is shown.
Replace {OBJECT}, {DAMAGE_CONFIG}, {VIEW} placeholders with real values.

## Environment check (do this first every session)
```bash
conda activate Reconstruction   # for Gemini calls (FullMethod.py, ai_view_judge.py)
# OR
conda activate Evaluation       # for cv2/matplotlib scripts
echo $GEMINI_API_KEY            # must print your real key, not empty
```

## MAIN BATCH PIPELINE (already complete for 10 objects, 160 experiments)
```bash
# Full reconstruction batch (resumable - skips anything already done)
python RunExperiment.py \
  --manifest configs/experiment_manifest.json \
  --blender_path "/Applications/Blender.app/Contents/MacOS/Blender" \
  --blender_render_script blender_render.py \
  --view_presets configs/view_presets.json \
  --output_root outputs \
  --full_method_script FullMethod.py \
  --objects_json objects.json \
  --damages_json damage.json

# Coregistration + eval batch (resumable, requires tapping through popup windows)
python CoregisterAndEvaluate.py \
  --manifest configs/experiment_manifest.json \
  --coregistration_script "/Users/stevenbu/Desktop/ImageCoregistration/image_register.py" \
  --extract_mask_script extract_missing_mask.py \
  --evaluate_script evaluate.py \
  --scale 1.0

# Aggregate all results into one CSV + console summary
python aggregate_metrics.py \
  --manifest configs/experiment_manifest.json \
  --output results/aggregated_metrics.csv

# Reset specific experiments to redo (doesn't touch renders/reconstructions)
python reset_evaluation.py \
  --manifest configs/experiment_manifest.json \
  --experiments {OBJECT}/{DAMAGE_CONFIG}/{VIEW}
```

## PER-EXPERIMENT MANUAL STEPS (for spot-checks, marker tests, judge tests)

### Standard coregistration + eval (from inside an experiment folder with input.png/ground_truth.png/generated.png):
```bash
mkdir -p data result
cp ground_truth.png data/ground_truth.png
cp generated.png data/generated.png   # or generated_smooth_outline.png etc - match filename below

python "/Users/stevenbu/Desktop/ImageCoregistration/image_register.py" ground_truth.png generated.png --sift --ransac --scale 1.0

# CHECK result/source_new.png and result/target_new.jpg visually before trusting numbers!
# Good alignment: reconstruction error roughly 8-40. Bad: 100+.
# Also check: is the bottle centered/upright, or tilted/cropped?

python /Users/stevenbu/Desktop/extract_missing_mask.py --reference ground_truth.png --comparison input.png --output GT_Mask.png --output_dir .
python /Users/stevenbu/Desktop/extract_missing_mask.py --reference result/source_new.png --comparison input.png --output GT_Fragment.png --output_dir .
python /Users/stevenbu/Desktop/evaluate.py --gt_mask GT_Mask.png --gt_fragment GT_Fragment.png --output_dir . --output_name metrics.json
python /Users/stevenbu/Desktop/centroid_distance.py --gt_mask GT_Mask.png --gt_fragment GT_Fragment.png --output_dir . --output_name centroid.json
```

### If SIFT alignment looks tilted/broken - manual fallback:
```bash
python "/Users/stevenbu/Desktop/manual_align_matplotlib.py" \
  --target ground_truth.png \
  --source generated.png \
  --coregistration_dir "/Users/stevenbu/Desktop/ImageCoregistration" \
  --scale 1.0
# Click 5-6 spread-out corresponding points on each image window, close window when done.
# Then continue with the extract_missing_mask/evaluate/centroid_distance steps above.
```

## SHAPE-DRAWING (RED CIRCLE) EXPERIMENT
From inside an already-rendered experiment folder (needs GT_Mask.png - run extract_missing_mask first if missing):
```bash
python /Users/stevenbu/Desktop/draw_marker.py \
  --gt_mask GT_Mask.png --input input.png \
  --output input_with_smooth_outline.png \
  --shape smooth_outline --color red

python /Users/stevenbu/Desktop/FullMethod.py \
  --metadata metadata.json \
  --objects /Users/stevenbu/Desktop/objects.json \
  --damages /Users/stevenbu/Desktop/damage.json \
  --marked_image input_with_smooth_outline.png \
  --use_marker_note --marker_hint_type shape \
  --output generated_smooth_outline.png

# CHECK generated_smooth_outline.png vs ground_truth.png visually before proceeding

# Then run coregistration + eval (see above), using generated_smooth_outline.png as source,
# and --output_name metrics_smooth_outline.json / centroid_smooth_outline.json to avoid
# overwriting baseline results in the same folder.
```

## AI VIEW JUDGE EXPERIMENT
```bash
# Step 1: render 5 candidate views (0/45/90/135/180 deg) for one object
# (separate output_root so it doesn't collide with main outputs/)
blender --background --python blender_render.py -- \
  --fragments_dir "{FULL_PATH_TO_OBJECT_FRACTURE_FOLDER}" \
  --object_id {OBJECT} \
  --fragment_map configs/fragment_maps/{OBJECT}.json \
  --view_presets configs/candidate_views.json \
  --output_root outputs_judge_test --engine WORKBENCH --resolution 1024 --distance_factor 1.5

# Step 2: run the judge (sends only input.png from each view - NEVER ground_truth)
python ai_view_judge.py \
  --object_id {OBJECT} \
  --damage_config {DAMAGE_CONFIG} \
  --views_root outputs_judge_test/{OBJECT}/{DAMAGE_CONFIG} \
  --objects objects.json \
  --damages damage.json \
  --output judge_choice_{OBJECT}_{DAMAGE_CONFIG}.json

# Step 3: reconstruct using the judge's chosen view (check judge_choice_*.json for chosen_view_id)
cd outputs_judge_test/{OBJECT}/{DAMAGE_CONFIG}/view_{CHOSEN_ID}
python /Users/stevenbu/Desktop/FullMethod.py \
  --metadata metadata.json \
  --objects /Users/stevenbu/Desktop/objects.json \
  --damages /Users/stevenbu/Desktop/damage.json \
  --output generated.png

# Then run coregistration + eval (see above) and compare against the fixed-view baseline
# for the same object/damage_config from the main outputs/ folder.
```

## OBJECT FRAGMENT PATHS (for --fragments_dir, from object_registry.json)
```
beer_bottle: /Users/stevenbu/Desktop/Breaking-Bad-Dataset.github.io/everyday/BeerBottle/3f91158956ad7db0322747720d7d37e8/fractured_29
wine_glass: /Users/stevenbu/Desktop/Breaking-Bad-Dataset.github.io/everyday/WineGlass/2d89d2b3b6749a9d99fbba385cc0d41d/fractured_55
vase: /Users/stevenbu/Desktop/Breaking-Bad-Dataset.github.io/everyday/Vase/1/fractured_66
wine_bottle: /Users/stevenbu/Desktop/Breaking-Bad-Dataset.github.io/everyday/WineBottle/1c79735033726294724d5ee7f09ab66b/fractured_30
bottle: /Users/stevenbu/Desktop/Breaking-Bad-Dataset.github.io/everyday/Bottle/1/fractured_61
spoon: /Users/stevenbu/Desktop/Breaking-Bad-Dataset.github.io/everyday/Spoon/1/fractured_26
ring: /Users/stevenbu/Desktop/Breaking-Bad-Dataset.github.io/everyday/Ring/2b0b9efff714f07883a2aadcb43f06a6/fractured_79
drink_bottle: /Users/stevenbu/Desktop/Breaking-Bad-Dataset.github.io/everyday/DrinkBottle/1ef68777bfdb7d6ba7a07ee616e34cd7/fractured_48
drinking_utensil: /Users/stevenbu/Desktop/Breaking-Bad-Dataset.github.io/everyday/DrinkingUtensil/2e228ee528f0a7054212ff51b27f0221/fractured_77
pill_bottle: /Users/stevenbu/Desktop/Breaking-Bad-Dataset.github.io/everyday/PillBottle/6aba175c988c8bec9fac57ed2e044778/fractured_14
```

## RESULTS SO FAR (see PROJECT_STATE.md for full narrative)
- Main batch: results/aggregated_metrics.csv (160 experiments, mean IoU 0.489)
- Combined 15 objects: results/combined_all_15_objects.csv (mean IoU 0.527)
- Marker experiment results: beer_bottle neck_simple (0.396->0.473), wine_bottle neck_simple (0.624->0.671), beer_bottle neck_severe (0.689->0.819, best result)
- AI Judge results: beer_bottle neck_simple (0.396->0.446), beer_bottle neck_severe (0.689->0.71-0.76, but with a concerning large centroid error at the judge-picked view - needs more investigation)
- Currently checking: does AI Judge always pick the same option regardless of object (positional bias check) - testing on wine_bottle now
