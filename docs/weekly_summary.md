# Weekly Summary — Fragment Repair Pipeline

## Goal this week

Following up on meeting notes, explored two proposed methods for improving VLM-based fragment reconstruction, tested them independently on real objects, and compiled results into a comparison table against the existing 160-experiment baseline.

## Methods tested

**1. AI View Judge** — instead of always reconstructing from a fixed camera angle, render 5 candidate views (0–180°, 45° increments, normal to the fracture direction) and have Gemini pick which view gives the best chance of a successful repair, then reconstruct from that chosen view.

**2. Red-Circle / Shape-Marker** — draw a smooth outline directly on the damaged input image showing exactly where the missing piece belongs, then reconstruct using that marked image as a guide.

**3. Verbal-Description Judge** (a variant of #1) — instead of letting the AI judge decide on its own, a human describes in their own words what makes a good view, and the AI picks whichever candidate best matches that description.

## Results

### Shape-Marker: 6 / 6 clean wins

| Object | Config | Baseline IoU | Marker IoU | Δ |
|---|---|---|---|---|
| beer_bottle | neck_simple | 0.396 | 0.473 | +0.077 |
| wine_bottle | neck_simple | 0.624 | 0.671 | +0.047 |
| beer_bottle | neck_severe | 0.689 | 0.819 | +0.130 |
| pill_bottle | body_severe | 0.526 | 0.544 | +0.018 |
| drink_bottle | neck_severe | 0.381 | 0.517 | +0.136 |
| **teapot** | **handle_simple** | **0.410** | **0.747** | **+0.337** |

Every single test showed improvement. Teapot's result is the standout — it had previously been considered a blocked, unmeasurable case due to a mask-extraction limitation on curved geometry, but the marker technique nearly doubled its baseline IoU and produced a visually excellent match to ground truth.

### AI View Judge: 1 strong win out of 4 tests

| Object | Config | Baseline (front) | Best baseline (any view) | Judge's pick | Judge IoU |
|---|---|---|---|---|---|
| beer_bottle | neck_simple | 0.396 | 0.778 | view_2 | 0.446 |
| beer_bottle | neck_severe | 0.689 | 0.822 | view_2 | 0.735 |
| wine_bottle | neck_severe | 0.874 | 0.929 | view_0 (= baseline's own view) | 0.781 |
| **vase** | **rim_severe** | **0.436** | **0.554** | **view_0** | **0.751** |

Vase is the clear highlight: the judge's pick beat even the best available baseline view by a wide margin, and was independently visually confirmed as a genuinely good choice.

### Verbal-Description Judge: tested, did not work reliably

Tested once on vase/rim_severe with a clear, correct human-written description. The model:
- Failed to pick the objectively correct view (the only one showing the complete missing piece) in **two separate attempts**
- Gave **two different wrong answers** across those two attempts with identical input — not just wrong, but self-inconsistent

This is treated as a confirmed negative result rather than an unresolved thread — a legitimate outcome of testing the idea properly.

## Full per-object baseline stats

All 15 objects now have mean, min, max, standard deviation, and standard error for both IoU and Dice, compiled into the results spreadsheet (10 objects from raw per-experiment data; 5 manual objects reconstructed using their original "prefer corrected value when SIFT failed" convention, verified to match previously reported means exactly).

## Open questions / honest limitations

- **AI Judge positional bias not fully ruled out**: 3 of 4 tests picked the first-shown candidate image. Vase's pick was independently confirmed as genuinely good, so content-sensitivity is real, but doesn't rule out bias stacking on top. A shuffled-image-order test would resolve this definitively.
- **Mask-extraction limitation on curved/thin fragments**: confirmed on teapot and pill_bottle — a cleanup tool was built and tested, works well on compact-blob fragments (wine_bottle) but not on thin/elongated ones or fully arc-shaped masks. IoU/Dice remain trustworthy regardless (they're cropped to the ground-truth region); centroid distance and area_ratio are the metrics affected.
- **Reconstruction variance discovered**: re-running the identical view/config through Gemini twice produced a real ~0.09 IoU swing, unrelated to any experimental variable. Single before/after comparisons should be read with this in mind.

## Suggested next steps

1. Shuffled candidate-image-order test to properly resolve the AI Judge positional-bias question
2. Broader sample size — most objects still only have 0–1 tests on the new methods; the 10 untouched objects (especially the trickier ones: ring, spoon, drinking_utensil) would strengthen the pattern
3. Repeated-run variance sampling on the marker/judge tracks specifically, now that this is known to matter
