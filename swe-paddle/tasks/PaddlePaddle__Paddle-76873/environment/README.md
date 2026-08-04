# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #76873 + follow-up #77103.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `471930236df5ba4e3bc34e1af6b8b9118e55a2d2` (parent of #76873 squash-merge `451e1af`)
- Gold endpoint: `231207ce894f7f13e5c68e24cfa251ad41d10532` (after #77103)
- Resource: CPU
- GPU required: no
- Patch type: **source build required**. Gold changes Python activation APIs,
  operator YAML config, and PIR symbolic-shape interfaces. A pure wheel overlay
  is not sufficient for the fixed state.
- Paddle install: checkout `base_commit` and build from source; after applying
  `solution/code.patch`, rebuild so YAML / C++ symbolic-shape changes take effect.

## Run Order (Run / Test / Fix)

1. Check out `PaddlePaddle/Paddle` at the base commit and build from source.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; inplace / compatibility F2P cases should
   **fail / error**. Existing non-inplace cases in the same modules should still
   pass where applicable (P2P candidates).
4. Apply `solution/code.patch` and **rebuild**.
5. Run `bash tests/test.sh` again; all target cases should **pass**.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Known Risks

- Source build / rebuild is mandatory for YAML and C++ symbolic-shape changes.
- Symbolic / CINN-related tests may need a richer environment; verifier can keep
  a stable F2P subset focused on legacy activation inplace tests if needed.
- `RReLU` and similar randomized paths need fixed seeds for stable node IDs.
- Gold patch is the **net** of #76873 and #77103 relative to the base commit
  (sequential PR diffs applied on base; not a raw `base..later-merge` tree diff).
