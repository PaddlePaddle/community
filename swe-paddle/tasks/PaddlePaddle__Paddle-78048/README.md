# PaddlePaddle__Paddle-78048

Source PR: https://github.com/PaddlePaddle/Paddle/pull/78048

This task covers parameter-alias compatibility for `paddle.hsplit`, `paddle.dsplit`, and `paddle.vsplit`.

- Base: `3f270c40db7776481d69176ee09222b3437d92bb`
- Gold: `e9f4d5fd4a0893b99b358b100383799ed52a0e7e`
- Production file: `python/paddle/tensor/manipulation.py`
- Upstream test file: `test/legacy_test/test_api_compatibility.py`

The task is Python-only and does not require rebuilding Paddle.

The F2P oracle uses the real tests added by the source PR:

- `TestHsplitAPI::test_dygraph_Compatibility`
- `TestDsplitAPI::test_dygraph_Compatibility`
- `TestVsplitAPI::test_dygraph_Compatibility`
