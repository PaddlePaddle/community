# 修复 `persistent_workers=True` 时提前结束迭代而导致 DataLoader 崩溃

## 详细描述

`paddle.io.DataLoader` 开启多进程加载和 `persistent_workers=True` 后，如果在遍历过程中使用 break 退出，再次迭代同一个 DataLoader 时报错。

例如：

```python
loader = paddle.io.DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
)

for epoch in range(3):
    for i, batch in enumerate(loader):
        if i > 10:
            break
```

运行到后续 epoch 时，报错信息如下：

```text
Traceback (most recent call last):
  File "test_dataloader.py", line 51, in <module>
    for i, (image, label) in enumerate(loader()):
  File "/usr/local/lib/python3.8/dist-packages/paddle/fluid/dataloader/dataloader_iter.py", line 746, in __next__
    data = _restore_batch(data, self._structure_infos.pop(0))
IndexError: pop from empty list
```

相同代码在 `persistent_workers=False` 时可以正常运行。

DataLoader 应支持在提前结束当前 epoch 后继续复用，后续迭代返回的 batch 结构和数据也应保持正确。

## 验收说明

* 开启 `persistent_workers=True` 时，可以在一个 epoch 中途结束迭代，并在下一个 epoch 继续使用同一个 DataLoader
* 提前结束并重新开始迭代时，不应出现 `IndexError: pop from empty list`
* 后续迭代返回的 batch 结构和数据应保持正确
* 完整迭代一个 epoch 的现有行为保持不变
* `persistent_workers=False` 时的现有行为保持不变

## 技术要求

* 熟悉 Python
* 了解 Paddle DataLoader 的多进程数据加载
* 了解 persistent workers 和 DataLoader iterator 的重置流程
