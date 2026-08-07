# Environment Notes

SWE-Paddle task candidate for PaddlePaddle/Paddle PR #59383 + follow-up #60835.

## Expected Environment

- Repository: `PaddlePaddle/Paddle`
- Base commit: `a8d5117371e8b9d16ff28011329bc04104eaf50a` (parent of #59383 merge `8d717f3`)
- Gold endpoint: `a92999d0788ab7d4241a3daf9cadcb67566ef541` (after #60835)
- Resource: CPU
- GPU required: no
- Patch type: **pure Python**. Gold changes public Tensor API exports and
  `masked_scatter` / `masked_scatter_` implementation only.
- Paddle install: prefer an era-matched (late 2023 / early 2024) wheel, or a
  source checkout at `base_commit` with importable Paddle. No C++ / codegen
  rebuild is required for the gold patch itself.

## Run Order (Run / Test / Fix)

1. Check out `PaddlePaddle/Paddle` at the base commit and ensure Paddle is importable.
2. Apply `tests/test.patch`.
3. Run `bash tests/test.sh`; `masked_scatter` F2P cases should **fail / error**.
   Existing unrelated inplace cases should still pass where applicable (P2P).
4. Apply `solution/code.patch`.
5. Run `bash tests/test.sh` again; all target cases should **pass**.

## Minimal Test Command

```bash
bash tests/test.sh
```

## Known Risks

- Historical wheels around 2023-12 may be hard to pin exactly; confirm
  `paddle.where` / basic dygraph+static APIs still match the exercised contracts.
- Gold is the **net** of #59383 and #60835 relative to the base commit
  (sequential PR diffs on base; not a raw `base..later-merge` tree diff).
