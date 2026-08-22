# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #52948 + follow-up #53572.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `cf6cbc347970a1fd2c9d76e427880139789497af` (parent of #52948 squash-merge `db30aa1`)
- Gold endpoint: `f3f3d57a159caf3b77f93a4d86cb233e6a1c159a` (after #53572)
- Resource: CPU
- GPU required: no
- Patch type: **pure Python**. Production changes are limited to static-graph /
  dy2static Python modules; no C++ / CUDA / kernel / infermeta rebuild is
  required for the gold patch itself.
- Paddle install: prefer an era-matched (2023-04/05) wheel or a source checkout
  at `base_commit` with a working Python package layout so that
  `paddle.jit.to_static` and static-graph execution are available.

## Run Order (Run / Test / Fix)

1. Check out `PaddlePaddle/Paddle` at the base commit and ensure Paddle is
   importable (era-matched wheel or source build/install).
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; the static / dy2static `register_hook` cases should
   **fail / error** before the fix. Existing dygraph-only hook cases should still
   pass (P2P candidates).
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all target cases should **pass**.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Known Risks

- Historical (2023-04) wheels may be hard to pin exactly; if using a newer
  wheel, confirm `to_static` / static Variable APIs still match the base-era
  contracts exercised by the tests.
- `test_hook_in_init_for_layer` uses random input; prefer fixed seeds when
  deriving stable F2P / P2P node IDs.
- Gold patch is the **net** of #52948 and #53572 relative to the base commit
  (intermediate helpers introduced in #52948 and removed/relocated in #53572
  are not left as dead intermediate state).
