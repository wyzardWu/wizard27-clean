# Data

## Training data

Training reads pre-encoded clips (121 frames at 30 fps, 832 × 480, Wan2.2 VAE latents of 31 × 30 × 52) under
`$WIZARD_DATA`:

```
wan_v7/          v7_rows.jsonl (one row per clip: group, key, paths), cond7/ (proxy latents, depth, tiers, cameras)
wan_mix5/        clips/ (video latents), text_table.pt (caption embeddings), neg_emb.pt (empty-prompt embedding)
wan_memroam/     roaming memory banks, mem_index*.pt, rep_keys_both30.json
wan_sfly/        revisit sequences (SceneFly), rows.jsonl
wan_sfly_replay/ revisit sequences replayed backwards, rows.jsonl
wan_lookback_geo/ synthetic look-back sequences (camera turns away and back), rows_full.jsonl
```

Each row belongs to a **group**, used for the mixing weights (`--group_rep group:weight,...`):

| Groups | Content |
|---|---|
| `e1o`, `e2o`, `e1n`, `e2n`, `e1x` | synthetic boss entries (one or two characters) |
| `turn` | turning while running |
| `roam` | roaming recordings |
| `sfly`, `sfly_replay`, `lookback`, `lookback0` | revisits: the camera comes back to a place it has seen |
| `ww`, `er`, `ue`, `bk` | duels |

The data-building pipeline and the datasets are not part of this repository.

## Visual proxy

One RGB frame per video frame (1024 × 576 before cropping):

| Channel | Meaning |
|---|---|
| R | world height |
| B | depth |
| G | tier: sky 0 · unknown ≈ 38 · coarse ≈ 77 · exact 153–255 (normal shading under a fixed light) |
| skeletons | drawn on top with a 2 px black outline; weapons and shields included |

Tiers come from the geometry's source: engine depth is *exact*, the first frame's MapAnything / MoGe-2 reconstruction is
*coarse*, everything outside the first view is *unknown*. Edits made by the world agent are *exact*.

## Inference case

A case directory holds the inputs of one shot:

| File | Content |
|---|---|
| `cond_v7.mkv` | the visual proxy (lossless FFV1) |
| `tier7.npy` | tier per token cell, `[T, 60, 104]` uint8 |
| `gbuf.npz` | per-frame grid depth, camera position / rotation / intrinsics, scene origin and player height |
| `world_px.npz` | per-pixel depth and tier (world maintenance) |
| `slots.npz` | character slot of every pixel (reference binding) |
| `first_frame.png` | the first frame |
| `prompt.txt` | the text prompt |

`tools/case_to_exact.py <case> <new case>` relabels a case's coarse geometry as exact (same geometry and cameras).
