### load_state_dict_from_url API 设计文档

| API 名称     | Load_state_dict_from_url                            |
| ------------ | --------------------------------------------------- |
| 提交作者     | zty-king                                            |
| 提交时间     | 2024-09-16                                          |
| 版本号       | V1.0                                                |
| 依赖飞桨版本 | develop                                             |
| 文件名       | 20240916_api_design_for_load_state_dict_from_url.md |

# 一、概述

## 1、相关背景

在深度学习和机器学习的研究和应用中，预训练模型提供了一种方便的方式来利用现有的训练成果，从而节省训练时间和计算资源。这些模型通常被存储为 PyTorch 的序列化对象，可以通过网络下载和加载。为了便于使用和管理这些模型，提升飞桨API丰富度，Paddle需要提供了一些工具和函数来简化从远程源加载模型的过程，即load_state_dict_from_url。

## 2、功能目标

`load_state_dict_from_url` 函数的主要功能是从指定的 URL 下载`Paddle`的模型权重（即 `state_dict`），并在必要时对下载的文件进行解压。它实现了以下目标：

1. **自动管理模型目录**：函数会检查并创建一个用于存储下载的模型文件的目录。如果用户没有指定目录，函数会使用默认目录。
2. **下载模型文件**：函数从指定的 URL 下载模型文件，并在下载过程中显示进度条（如果设置了）。
3. **文件哈希验证**：函数支持对下载的文件进行哈希验证，以确保文件的完整性和唯一性。
4. **解压支持**：如果下载的文件是一个 zip 文件，函数会自动解压。
5. **加载模型**：函数会将下载的模型文件加载到 Paddle中，并处理旧格式的文件（如 zip 文件）。

## 3、意义

为 `Paddle.hub` 增加 `load_state_dict_from_url` ，丰富 `Paddle.hub` 中模型权重加载相关的 API。

# 二、飞桨现状

`Paddle` 目前没有提供直接下载或者加载模型权重文件的方法。

# 三、业内方案调研

## PyTorch

`Pytorch` 底层实现了模型权重的加载与下载功能，在 `Python` 端， `PyTorch.hub.load_state_dict_from_url` 函数直接实现上述功能，因此参考PyTorch中的函数实现方法，来开发`Paddle.hub.load_state_dict_from_url`API

### API 文档

- [torch.hub — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/hub.html#torch.hub.load_state_dict_from_url)
    - Parameters
        - **url** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)) – URL of the object to download
        - **model_dir** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) – directory in which to save the object
        - **map_location** (*optional*) – a function or a dict specifying how to remap storage locations (see torch.load)
        - **progress** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – whether or not to display a progress bar to stderr. Default: True
        - **check_hash** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If True, the filename part of the URL should follow the naming convention `filename-<sha256>.ext` where `<sha256>` is the first eight or more digits of the SHA256 hash of the contents of the file. The hash is used to ensure unique names and to verify the contents of the file. Default: False
        - **file_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str)*,* *optional*) – name for the downloaded file. Filename from `url` will be used if not set.
        - **weights_only** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – If True, only weights will be loaded and no complex pickled objects. Recommended for untrusted sources. See [`load()`](https://pytorch.org/docs/stable/generated/torch.load.html#torch.load) for more details.
    - Return type
        - [*Dict*](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]

### 实现逻辑 

#### `Python` 端

关键源码

- [pytorch/torch/hub.py at main · pytorch/pytorch (github.com)](https://github.com/pytorch/pytorch/blob/main/torch/hub.py)

```Python
def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None, weights_only=False):
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location)
    return torch.load(cached_file, map_location=map_location)
```

#### CPU端

`PyTorch` 实现。

#### GPU端

`PyTorch` 实现。

## TensorFlow

- **TensorFlow Hub**: TensorFlow Hub 提供了一个模型仓库，用户可以方便地从远程下载并加载预训练模型。例如，使用 `tensorflow_hub` 包：

```
import tensorflow_hub as hub

# 加载预训练模型
model = hub.load("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4")
```

- **Keras Applications**: Keras 的 `applications` 模块允许用户从内置模型库中加载预训练模型。例如：

  ```
  from tensorflow.keras.applications import MobileNetV2
  
  # 加载预训练的 MobileNetV2 模型
  model = MobileNetV2(weights='imagenet')
  ```

## MXNet

**MXNet**提供了类似的功能，通常通过其 `gluoncv` 模块来实现，gluoncv 提供了丰富的预训练模型和加载功能。例如：

```
from gluoncv import model_zoo

# 下载并加载预训练模型
net = model_zoo.get_model('mobilenet1.0', pretrained=True)
```

# 四、对比分析

目前，主流深度学习框架 `Pytorch` 实现了该方法，并且比较符合`Paddle`当前的需求，即从给定 URL 加载 Paddle 序列化对象。如果下载的文件是 zip 文件，它将自动解压缩。如果对象已存在于 model_dir 中，则将其反序列化并返回，因此直接以`Pytorch`的load_state_dict_from_url函数为模板开发Paddle的API即可。

# 五、设计思路与实现方案
## 命名与参数设计

### 添加 Python API:

```
Paddle.hub.load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None, weights_only=False)
```

### 参数表

| 参数名       | 类型             | 表述                                                         |
| ------------ | ---------------- | ------------------------------------------------------------ |
| url          | string           | 要下载的对象的 URL。                                         |
| model_dir    | string, optional | 保存了下载对象的目录。                                       |
| map_location | optional         | 一个函数或字典，用于指定如何重新映射存储位置。               |
| progress     | bool, optional   | 是否显示进度条到标准错误输出。默认值：`True`。               |
| check_hash   | bool, optional   | 如果为 `True`，URL 的文件名部分应遵循命名约定 `filename-<sha256>.ext`，其中 `<sha256>` 是文件内容的 SHA256 哈希的前八位或更多位。哈希用于确保唯一的名称和验证文件内容。默认值：`False`。 |
| file_name    | string, optional | 下载文件的名称。如果未设置，将使用 URL 中的文件名。          |
| weights_only | bool, optional   | 如果为 True，则只会加载权重，而不会加载复杂的序列化对象。建议用于不受信任的来源。 |

## 底层 OP 设计

不涉及底层OP。

## API实现方案

- **load_state_dict_from_url**的函数实现：

```python
def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None, weights_only=False):
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location)
    return paddle.hub.load(cached_file, map_location=map_location)
```

- 设置check_hash中的正则表达式：

```
# matches bfd8deac from resnet18-bfd8deac.pth
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
```

- 函数**download_url_to_file**根据url下载文件到本地，`Paddle/PaddleMIX/paddlemix/datacopilot/misc/_download.py`有download_url_to_file函数的实现方法，直接使用或集成到hub.py

```
def download_url_to_file(url: str, dst: str, hash_prefix: Optional[str] = None, progress: bool = True) -> None:
    r"""Download object at the given URL to a local path.
    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Reference: 
        https://github.com/pytorch/pytorch/blob/main/torch/hub.py
    """
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    # We deliberately do not use NamedTemporaryFile to avoid restrictive
    # file permissions being applied to the downloaded file.
    dst = os.path.expanduser(dst)
    for seq in range(tempfile.TMP_MAX):
        tmp_dst = dst + '.' + uuid.uuid4().hex + '.partial'
        try:
            f = open(tmp_dst, 'w+b')
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, 'No usable temporary file name found')

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(128 * 1024)
                if len(buffer) == 0:
                    break
                f.write(buffer)  # type: ignore[possibly-undefined]
                if hash_prefix is not None:
                    sha256.update(buffer)  # type: ignore[possibly-undefined]
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()  # type: ignore[possibly-undefined]
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError(f'invalid hash value (expected "{hash_prefix}", got "{digest}")')
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

```

- 函数 **_is_legacy_zip_format** 判断是否为ZIP文件的函数：

```python
def _is_legacy_zip_format(filename):
    if zipfile.is_zipfile(filename):
        infolist = zipfile.ZipFile(filename).infolist()
        return len(infolist) == 1 and not infolist[0].is_dir()
    return False
```

- 函数 **_legacy_zip_load** ZIP文件解压，并用load函数加载文件：

```python
def _legacy_zip_load(filename, model_dir, map_location):
    warnings.warn('Falling back to the old format < 1.6. This support will be '
                  'deprecated in favor of default zipfile format introduced in 1.6. '
                  'Please redo paddle.save() to save it in the new zipfile format.')
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        if len(members) != 1:
            raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
        f.extractall(model_dir)
        extraced_name = members[0].filename
        extracted_file = os.path.join(model_dir, extraced_name)
    return paddle.hub.load(extracted_file, map_location=map_location)
```

- 函数 **get_dir()** 获取 `Paddle Hub`缓存目录的路径

```
def get_dir():
    if os.getenv('PADDLE_HUB'):
        warnings.warn('PADDLE_HUB is deprecated, please use env PADDLE_HOME instead')

    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(_get_paddle_home(), 'hub')

def _get_paddle_home():
    # Get the Paddle home directory from the environment variable or default to a standard location
    paddle_home = os.path.expanduser(
        os.getenv('PADDLE_HOME',
                  os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'paddle')))
    return paddle_home
```



# 六、测试和验收的考量

测试考虑的case如下：

    1.用Paddle.hub.load_state_dict_from_url()加载url，下载模型权重；同时手动下载对应url的多个模型权重文件，用paddle.hub.load()加载文件，进行结果对齐；
    
    2.用Paddle.hub.load_state_dict_from_url()加载url，下载压缩的模型权重，即ZIP格式文件；同时手动下载对应url的多个模型权重ZIP文件，并手动解压，用paddle.hub.load()加载文件，进行结果对齐；
    
    3.用Paddle.hub.load_state_dict_from_url()加载已经下载的模型权重文件；同时用paddle.hub.load()加载对应的模型权重文件，进行结果对齐；

- **硬件场景**
  覆盖 CPU、GPU 两种测试场景

- **输出正确性**
  输出数值结果的一致性和数据类型是否正确


# 七、可行性分析及规划排期

- 第一周，实现相关代码
- 第二周，测试用例和文档
- 第三周，Review

# 八、影响面

新增 API，对其他模块无影响。

# 名词解释

无

# 附件及参考资料

- [torch.hub — PyTorch 2.4 documentation](https://pytorch.org/docs/stable/hub.html#torch.hub.load_state_dict_from_url)
- [load-API文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/hub/load_cn.html)
