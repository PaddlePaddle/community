# paddle.utils.cpp_extension 基于 Setuptools 80+ 版本自定义算子机制适配设计文档

| API名称 | paddle.utils.cpp_extension 基于 Setuptools 80+ 版本自定义算子机制适配设计文档 |
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | megemini |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2025-10-30 |
|版本号 | V1.0 |
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 |
|文件名 | 20251030_api_design_for_setuptools80_custom_operator.md<br> |


# 一、概述

## 1、相关背景

关联任务：https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_9th/%E3%80%90Hackathon_9th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no109-基于-setuptools-80-版本自定义算子机制适配

关联 PR：https://github.com/PaddlePaddle/Paddle/pull/76008

PaddlePaddle 目前对于自定义 C++ 算子的实现是基于 setuptools 做了一些 patch，在 bdist_egg 阶段通过 patch write_stub 实现的，然而在 setuptools 80+，被 patch 的逻辑在 install command 不会被走到（于 pypa/setuptools#2908 移除），因此我们希望基于 setuptools 80+ 对自定义 C++ 算子进行适配，确保自定义 C++ 算子在 setuptools 80+ 是可用的。

## 2、功能目标

适配 Setuptools 80+ 版本，确保自定义算子在新版本 Setuptools 下能够正常编译、安装和运行，并保持向后兼容性，确保在旧版本 Setuptools 下仍能正常工作。

## 3、意义

确保自定义算子在新版本 Setuptools 下能够正常编译、安装和运行。

# 二、飞桨现状

PaddlePaddle 目前的自定义算子机制主要通过 `paddle.utils.cpp_extension` 模块实现，依赖 `write_stub` 机制生成 Python API 文件，该机制在 Setuptools 80+ 中不再自动触发。

# 三、业内方案调研

不涉及

# 四、对比分析

不涉及

# 五、设计思路与实现方案

## 命名与参数设计

本次改进不涉及新增 API，主要是对现有 `paddle.utils.cpp_extension.setup` 函数的内部实现进行改进。

## 底层OP设计

不涉及。

## API实现方案

首先，目前的实现方案，使用 `pip install .` 安装后无法运行 (setuptools < 80)，以下是测试日志：

``` shell

# 检查 setuptools 的版本
➜  tmp_setuptools git:(setuptools80) ✗ pip show setuptools
Name: setuptools
Version: 57.1.0
Summary: Easily download, build, install, upgrade, and uninstall Python packages
Home-page: https://github.com/pypa/setuptools
Author: Python Packaging Authority
Author-email: distutils-sig@python.org
License: UNKNOWN
Location: /usr/local/lib/python3.9/dist-packages
Requires:
Required-by: astroid, nodeenv, wandb

# 使用 pip install . 安装自定义算子
➜  tmp_setuptools git:(setuptools80) ✗ pip install .
Looking in indexes: https://mirrors.aliyun.com/pypi/simple/
Processing /paddle/Paddle/build/tmp_setuptools
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: custom-setup-ops
  Building wheel for custom-setup-ops (setup.py) ... done
  Created wheel for custom-setup-ops: filename=custom_setup_ops-0.0.0-cp39-cp39-linux_x86_64.whl size=1215482 sha256=7df3c5f2d60a213c810c649c0e6b0ae47380a2c36f7059e6aa849c2d194e183e
  Stored in directory: /tmp/pip-ephem-wheel-cache-izcix4g2/wheels/f4/9a/9b/ce79fe326a8ea140a10a9d2f0015460ed9eeadd188b749ac46
Successfully built custom-setup-ops
Installing collected packages: custom-setup-ops
Successfully installed custom-setup-ops-0.0.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

# 测试自定义算子，虽然安装成功，但是运行失败
➜  tmp_setuptools git:(setuptools80) ✗ python test.py
Traceback (most recent call last):
  File "/paddle/Paddle/build/tmp_setuptools/test.py", line 2, in <module>
    from custom_setup_ops import custom_relu
ImportError: dynamic module does not define module export function (PyInit_custom_setup_ops)

# 查看当前目录，清理掉生成的文件
➜  tmp_setuptools git:(setuptools80) ✗ l
total 36K
drwxr-xr-x  4 1000 1000 4.0K Nov  3 06:35 .
drwxr-xr-x 11 1000 1000 4.0K Oct 22 06:08 ..
drwxr-xr-x  3 root root 4.0K Nov  3 06:35 build
drwxr-xr-x  2 root root 4.0K Nov  3 06:35 custom_setup_ops.egg-info
-rw-rw-r--  1 1000 1000  464 Oct 21 10:07 readme.md
-rw-rw-r--  1 1000 1000 2.5K Oct 21 14:04 relu_cpu.cc
-rw-r--r--  1 root root  166 Oct 30 07:40 setup.py
-rw-rw-r--  1 1000 1000  166 Oct 21 14:06 setup_cpu.py
-rw-rw-r--  1 1000 1000  173 Oct 23 07:27 test.py
➜  tmp_setuptools git:(setuptools80) ✗ rm -rf build custom_setup_ops.egg-info

# 卸载
➜  tmp_setuptools git:(setuptools80) ✗ pip uninstall custom_setup_ops
Found existing installation: custom-setup-ops 0.0.0
Uninstalling custom-setup-ops-0.0.0:
  Would remove:
    /usr/local/lib/python3.9/dist-packages/custom_setup_ops-0.0.0.dist-info/*
    /usr/local/lib/python3.9/dist-packages/custom_setup_ops.so
    /usr/local/lib/python3.9/dist-packages/version.txt
Proceed (Y/n)? y
  Successfully uninstalled custom-setup-ops-0.0.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
➜  tmp_setuptools git:(setuptools80) ✗ l
total 28K
drwxr-xr-x  2 1000 1000 4.0K Nov  3 06:35 .
drwxr-xr-x 11 1000 1000 4.0K Oct 22 06:08 ..
-rw-rw-r--  1 1000 1000  464 Oct 21 10:07 readme.md
-rw-rw-r--  1 1000 1000 2.5K Oct 21 14:04 relu_cpu.cc
-rw-r--r--  1 root root  166 Oct 30 07:40 setup.py
-rw-rw-r--  1 1000 1000  166 Oct 21 14:06 setup_cpu.py
-rw-rw-r--  1 1000 1000  173 Oct 23 07:27 test.py

# 使用 setup.py 安装，安装成功，运行测试也可以运行
➜  tmp_setuptools git:(setuptools80) ✗ python setup.py install
running install
running bdist_egg
running egg_info
creating custom_setup_ops.egg-info
writing custom_setup_ops.egg-info/PKG-INFO
writing dependency_links to custom_setup_ops.egg-info/dependency_links.txt
writing top-level names to custom_setup_ops.egg-info/top_level.txt
writing manifest file 'custom_setup_ops.egg-info/SOURCES.txt'
reading manifest file 'custom_setup_ops.egg-info/SOURCES.txt'
writing manifest file 'custom_setup_ops.egg-info/SOURCES.txt'
installing library code to build/custom_setup_ops/bdist.linux-x86_64/egg
running install_lib
running build_ext
Compiling user custom op, it will cost a few seconds.....
building 'custom_setup_ops' extension
creating /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build
creating /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops
creating /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops/temp.linux-x86_64-3.9
/usr/local/bin/ccache x86_64-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/lib/python3.9/dist-packages/paddle/include -I/usr/local/lib/python3.9/dist-packages/paddle/include/third_party -I/usr/local/lib/python3.9/dist-packages/paddle/include/paddle/phi/api/include/compat -I/usr/local/lib/python3.9/dist-packages/paddle/include/paddle/phi/api/include/compat/torch/csrc/api/include -I/usr/include/python3.9 -I/usr/include/python3.9 -c /paddle/Paddle/build/tmp_setuptools/relu_cpu.cc -o /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops/temp.linux-x86_64-3.9/relu_cpu.o -w -DPADDLE_WITH_CUSTOM_KERNEL -DPADDLE_EXTENSION_NAME=custom_setup_ops -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++17
/paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops/temp.linux-x86_64-3.9/relu_cpu.o is compiled
x86_64-linux-gnu-g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -g -fwrapv -O2 -Wl,-Bsymbolic-functions -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops/temp.linux-x86_64-3.9/relu_cpu.o -L/usr/local/lib/python3.9/dist-packages/paddle/libs -L/usr/local/lib/python3.9/dist-packages/paddle/base -Wl,--enable-new-dtags,-R/usr/local/lib/python3.9/dist-packages/paddle/libs -Wl,--enable-new-dtags,-R/usr/local/lib/python3.9/dist-packages/paddle/base -o build/custom_setup_ops/lib.linux-x86_64-3.9/custom_setup_ops.so -l:libpaddle.so
Removed: build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops/temp.linux-x86_64-3.9/relu_cpu.o
creating build/custom_setup_ops/bdist.linux-x86_64
creating build/custom_setup_ops/bdist.linux-x86_64/egg
creating build/custom_setup_ops/bdist.linux-x86_64/egg/build
creating build/custom_setup_ops/bdist.linux-x86_64/egg/build/custom_setup_ops
creating build/custom_setup_ops/bdist.linux-x86_64/egg/build/custom_setup_ops/temp.linux-x86_64-3.9
copying build/custom_setup_ops/lib.linux-x86_64-3.9/version.txt -> build/custom_setup_ops/bdist.linux-x86_64/egg
copying build/custom_setup_ops/lib.linux-x86_64-3.9/custom_setup_ops.so -> build/custom_setup_ops/bdist.linux-x86_64/egg
creating stub loader for custom_setup_ops.so
Received len(custom_op) = 1, using custom operator
byte-compiling build/custom_setup_ops/bdist.linux-x86_64/egg/custom_setup_ops.py to custom_setup_ops.cpython-39.pyc
creating build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
copying custom_setup_ops.egg-info/PKG-INFO -> build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
copying custom_setup_ops.egg-info/SOURCES.txt -> build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
copying custom_setup_ops.egg-info/dependency_links.txt -> build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
copying custom_setup_ops.egg-info/not-zip-safe -> build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
copying custom_setup_ops.egg-info/top_level.txt -> build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO
writing build/custom_setup_ops/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
creating dist
creating 'dist/custom_setup_ops-0.0.0-py3.9-linux-x86_64.egg' and adding 'build/custom_setup_ops/bdist.linux-x86_64/egg' to it
removing 'build/custom_setup_ops/bdist.linux-x86_64/egg' (and everything under it)
Processing custom_setup_ops-0.0.0-py3.9-linux-x86_64.egg
creating /usr/local/lib/python3.9/dist-packages/custom_setup_ops-0.0.0-py3.9-linux-x86_64.egg
Extracting custom_setup_ops-0.0.0-py3.9-linux-x86_64.egg to /usr/local/lib/python3.9/dist-packages
Adding custom-setup-ops 0.0.0 to easy-install.pth file

Installed /usr/local/lib/python3.9/dist-packages/custom_setup_ops-0.0.0-py3.9-linux-x86_64.egg
Processing dependencies for custom-setup-ops==0.0.0
Finished processing dependencies for custom-setup-ops==0.0.0

# 测试
➜  tmp_setuptools git:(setuptools80) ✗ python test.py
Tensor(shape=[4, 10], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[0.        , 0.        , 0.27623373, 0.        , 0.33783096, 0.        ,
         0.        , 0.        , 1.96703553, 0.        ],
        [1.35411644, 1.67994177, 0.74918306, 1.07646525, 0.        , 0.        ,
         1.62951136, 0.        , 0.        , 0.        ],
        [0.90654951, 0.        , 0.        , 0.85921216, 0.36175427, 0.18975830,
         2.39137697, 0.        , 0.        , 0.00654856],
        [1.14674675, 1.73321831, 1.14670050, 0.        , 0.        , 0.47941723,
         0.04274137, 0.58359218, 0.84339291, 0.        ]])
➜  tmp_setuptools git:(setuptools80) ✗ l
total 40K
drwxr-xr-x  5 1000 1000 4.0K Nov  3 06:36 .
drwxr-xr-x 11 1000 1000 4.0K Oct 22 06:08 ..
drwxr-xr-x  3 root root 4.0K Nov  3 06:36 build
drwxr-xr-x  2 root root 4.0K Nov  3 06:36 custom_setup_ops.egg-info
drwxr-xr-x  2 root root 4.0K Nov  3 06:36 dist
-rw-rw-r--  1 1000 1000  464 Oct 21 10:07 readme.md
-rw-rw-r--  1 1000 1000 2.5K Oct 21 14:04 relu_cpu.cc
-rw-r--r--  1 root root  166 Oct 30 07:40 setup.py
-rw-rw-r--  1 1000 1000  166 Oct 21 14:06 setup_cpu.py
-rw-rw-r--  1 1000 1000  173 Oct 23 07:27 test.py

# 清理安装的文件，
➜  tmp_setuptools git:(setuptools80) ✗ rm -rf build custom_setup_ops.egg-info dist
➜  tmp_setuptools git:(setuptools80) ✗ pip uninstall custom_setup_ops
Found existing installation: custom-setup-ops 0.0.0
Uninstalling custom-setup-ops-0.0.0:
  Would remove:
    /usr/local/lib/python3.9/dist-packages/custom_setup_ops-0.0.0-py3.9-linux-x86_64.egg
Proceed (Y/n)? y
  Successfully uninstalled custom-setup-ops-0.0.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

# 卸载是干净的
➜  tmp_setuptools git:(setuptools80) ✗ ls -al /usr/local/lib/python3.9/dist-packages | grep custom
➜  tmp_setuptools git:(setuptools80) ✗

```

因此，需要增加 `pip install .` 的兼容代码：

``` python

    # Compatible with wheel installation via `pip install .`
    # Note: This is rarely used with modern pip, which uses bdist_wheel instead
    assert 'install' not in cmdclass
    cmdclass['install'] = InstallCommand

...

    def run(self):
        super().run()

        # Compatible with wheel installation via `pip install .`
        self._generate_python_api_file()

        self._clean_intermediate_files()

```

以下是测试结果：

``` shell

# 查看 setuptools 版本
➜  tmp_setuptools git:(setuptools80) ✗ pip show setuptools
Name: setuptools
Version: 57.1.0
Summary: Easily download, build, install, upgrade, and uninstall Python packages
Home-page: https://github.com/pypa/setuptools
Author: Python Packaging Authority
Author-email: distutils-sig@python.org
License: UNKNOWN
Location: /usr/local/lib/python3.9/dist-packages
Requires:
Required-by: astroid, nodeenv, wandb

# 使用 pip install . 进行安装
➜  tmp_setuptools git:(setuptools80) ✗ pip install .
Looking in indexes: https://mirrors.aliyun.com/pypi/simple/
Processing /paddle/Paddle/build/tmp_setuptools
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: custom-setup-ops
  Building wheel for custom-setup-ops (setup.py) ... done
  Created wheel for custom-setup-ops: filename=custom_setup_ops-0.0.0-cp39-cp39-linux_x86_64.whl size=1216623 sha256=91a4b5dad27ad457c96a1433a58de5499b3de197e99cc478a281ca9d9a6eac2e
  Stored in directory: /tmp/pip-ephem-wheel-cache-h3o8wps2/wheels/f4/9a/9b/ce79fe326a8ea140a10a9d2f0015460ed9eeadd188b749ac46
Successfully built custom-setup-ops
Installing collected packages: custom-setup-ops
Successfully installed custom-setup-ops-0.0.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv

# 测试通过
➜  tmp_setuptools git:(setuptools80) ✗ python test.py
Tensor(shape=[4, 10], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[0.61926198, 0.00549045, 1.72637534, 0.        , 0.12566043, 2.02829361,
         0.86272985, 0.        , 0.        , 0.        ],
        [0.28616923, 0.        , 0.51520216, 1.60348010, 0.        , 0.08020544,
         1.21574390, 0.        , 0.01430061, 0.        ],
        [0.        , 0.18555178, 1.01460934, 0.        , 0.35284781, 0.        ,
         1.40650642, 0.        , 0.73371738, 0.        ],
        [0.74932104, 0.27119094, 0.        , 1.08290946, 0.        , 0.        ,
         0.        , 0.99430609, 0.23505895, 0.51908028]])
➜  tmp_setuptools git:(setuptools80) ✗ l
total 36K
drwxr-xr-x  4 1000 1000 4.0K Nov  3 06:40 .
drwxr-xr-x 11 1000 1000 4.0K Oct 22 06:08 ..
drwxr-xr-x  3 root root 4.0K Nov  3 06:40 build
drwxr-xr-x  2 root root 4.0K Nov  3 06:40 custom_setup_ops.egg-info
-rw-rw-r--  1 1000 1000  464 Oct 21 10:07 readme.md
-rw-rw-r--  1 1000 1000 2.5K Oct 21 14:04 relu_cpu.cc
-rw-r--r--  1 root root  166 Oct 30 07:40 setup.py
-rw-rw-r--  1 1000 1000  166 Oct 21 14:06 setup_cpu.py
-rw-rw-r--  1 1000 1000  173 Oct 23 07:27 test.py
➜  tmp_setuptools git:(setuptools80) ✗ rm -rf build custom_setup_ops.egg-info

# 卸载是干净的
➜  tmp_setuptools git:(setuptools80) ✗ pip uninstall custom_setup_ops
Found existing installation: custom-setup-ops 0.0.0
Uninstalling custom-setup-ops-0.0.0:
  Would remove:
    /usr/local/lib/python3.9/dist-packages/custom_setup_ops-0.0.0.dist-info/*
    /usr/local/lib/python3.9/dist-packages/custom_setup_ops/*
    /usr/local/lib/python3.9/dist-packages/version.txt
Proceed (Y/n)? y
  Successfully uninstalled custom-setup-ops-0.0.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
➜  tmp_setuptools git:(setuptools80) ✗ ls -al /usr/local/lib/python3.9/dist-packages | grep custom

# 使用 python setup.py install 进行安装
➜  tmp_setuptools git:(setuptools80) ✗ python setup.py install
running install
running build
running build_ext
Compiling user custom op, it will cost a few seconds.....
building 'custom_setup_ops' extension
creating /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build
creating /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops
creating /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops/temp.linux-x86_64-3.9
/usr/local/bin/ccache x86_64-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/local/lib/python3.9/dist-packages/paddle/include -I/usr/local/lib/python3.9/dist-packages/paddle/include/third_party -I/usr/local/lib/python3.9/dist-packages/paddle/include/paddle/phi/api/include/compat -I/usr/local/lib/python3.9/dist-packages/paddle/include/paddle/phi/api/include/compat/torch/csrc/api/include -I/usr/include/python3.9 -I/usr/include/python3.9 -c /paddle/Paddle/build/tmp_setuptools/relu_cpu.cc -o /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops/temp.linux-x86_64-3.9/relu_cpu.o -w -DPADDLE_WITH_CUSTOM_KERNEL -DPADDLE_EXTENSION_NAME=custom_setup_ops -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++17
/paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops/temp.linux-x86_64-3.9/relu_cpu.o is compiled
x86_64-linux-gnu-g++ -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -g -fwrapv -O2 -Wl,-Bsymbolic-functions -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops/temp.linux-x86_64-3.9/relu_cpu.o -L/usr/local/lib/python3.9/dist-packages/paddle/libs -L/usr/local/lib/python3.9/dist-packages/paddle/base -Wl,--enable-new-dtags,-R/usr/local/lib/python3.9/dist-packages/paddle/libs -Wl,--enable-new-dtags,-R/usr/local/lib/python3.9/dist-packages/paddle/base -o build/custom_setup_ops/lib.linux-x86_64-3.9/custom_setup_ops.so -l:libpaddle.so
Received len(custom_op) = 1, using custom operator
Removed: build/custom_setup_ops/lib.linux-x86_64-3.9/build/custom_setup_ops/temp.linux-x86_64-3.9/relu_cpu.o
running install_lib
copying build/custom_setup_ops/lib.linux-x86_64-3.9/version.txt -> /usr/local/lib/python3.9/dist-packages
copying build/custom_setup_ops/lib.linux-x86_64-3.9/custom_setup_ops.py -> /usr/local/lib/python3.9/dist-packages
copying build/custom_setup_ops/lib.linux-x86_64-3.9/custom_setup_ops.so -> /usr/local/lib/python3.9/dist-packages
byte-compiling /usr/local/lib/python3.9/dist-packages/custom_setup_ops.py to custom_setup_ops.cpython-39.pyc
running install_egg_info
running egg_info
creating custom_setup_ops.egg-info
writing custom_setup_ops.egg-info/PKG-INFO
writing dependency_links to custom_setup_ops.egg-info/dependency_links.txt
writing top-level names to custom_setup_ops.egg-info/top_level.txt
writing manifest file 'custom_setup_ops.egg-info/SOURCES.txt'
reading manifest file 'custom_setup_ops.egg-info/SOURCES.txt'
writing manifest file 'custom_setup_ops.egg-info/SOURCES.txt'
Copying custom_setup_ops.egg-info to /usr/local/lib/python3.9/dist-packages/custom_setup_ops-0.0.0-py3.9.egg-info
running install_scripts

# 测试通过
➜  tmp_setuptools git:(setuptools80) ✗ python test.py
Tensor(shape=[4, 10], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[0.        , 2.07537651, 0.        , 0.        , 0.37558576, 0.        ,
         0.27504009, 0.25345257, 0.        , 0.        ],
        [0.00597949, 0.        , 0.        , 0.30615386, 0.        , 0.32855675,
         0.        , 0.05931013, 0.40368199, 0.        ],
        [0.        , 0.78842884, 0.        , 0.        , 0.04574476, 1.54774368,
         1.50367701, 0.        , 0.        , 0.        ],
        [0.        , 0.        , 0.        , 0.        , 0.        , 0.50843334,
         0.        , 0.59568095, 0.        , 0.24087408]])
➜  tmp_setuptools git:(setuptools80) ✗ l
total 36K
drwxr-xr-x  4 1000 1000 4.0K Nov  3 06:41 .
drwxr-xr-x 11 1000 1000 4.0K Oct 22 06:08 ..
drwxr-xr-x  3 root root 4.0K Nov  3 06:41 build
drwxr-xr-x  2 root root 4.0K Nov  3 06:41 custom_setup_ops.egg-info
-rw-rw-r--  1 1000 1000  464 Oct 21 10:07 readme.md
-rw-rw-r--  1 1000 1000 2.5K Oct 21 14:04 relu_cpu.cc
-rw-r--r--  1 root root  166 Oct 30 07:40 setup.py
-rw-rw-r--  1 1000 1000  166 Oct 21 14:06 setup_cpu.py
-rw-rw-r--  1 1000 1000  173 Oct 23 07:27 test.py
➜  tmp_setuptools git:(setuptools80) ✗ rm -rf build custom_setup_ops.egg-info

# 卸载是干净的
➜  tmp_setuptools git:(setuptools80) ✗ pip uninstall custom_setup_ops
Found existing installation: custom-setup-ops 0.0.0
Uninstalling custom-setup-ops-0.0.0:
  Would remove:
    /usr/local/lib/python3.9/dist-packages/custom_setup_ops
    /usr/local/lib/python3.9/dist-packages/custom_setup_ops-0.0.0-py3.9.egg-info
Proceed (Y/n)? y
  Successfully uninstalled custom-setup-ops-0.0.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
➜  tmp_setuptools git:(setuptools80) ✗ ls -al /usr/local/lib/python3.9/dist-packages | grep custom
➜  tmp_setuptools git:(setuptools80) ✗


```

测试 setuptools > 80 :

``` shell

➜  tmp_setuptools git:(setuptools80) ✗ pip show setuptools
Name: setuptools
Version: 80.9.0
Summary: Easily download, build, install, upgrade, and uninstall Python packages
Home-page:
Author:
Author-email: Python Packaging Authority <distutils-sig@python.org>
License:
Location: /usr/local/lib/python3.9/dist-packages
Requires:
Required-by: astroid, nodeenv, wandb
➜  tmp_setuptools git:(setuptools80) ✗ pip install .
Looking in indexes: https://mirrors.aliyun.com/pypi/simple/
Processing /paddle/Paddle/build/tmp_setuptools
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: custom_setup_ops
  Building wheel for custom_setup_ops (setup.py) ... done
  Created wheel for custom_setup_ops: filename=custom_setup_ops-0.0.0-cp39-cp39-linux_x86_64.whl size=1216524 sha256=c4c41405f22101de7a61b53a70a07c9fa3f7240aabd0b8991ec655d05c502482
  Stored in directory: /tmp/pip-ephem-wheel-cache-6p8r9p2p/wheels/f4/9a/9b/ce79fe326a8ea140a10a9d2f0015460ed9eeadd188b749ac46
Successfully built custom_setup_ops
Installing collected packages: custom_setup_ops
Successfully installed custom_setup_ops-0.0.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
➜  tmp_setuptools git:(setuptools80) ✗ python test.py
Tensor(shape=[4, 10], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[0.73605037, 0.        , 2.10240412, 0.22792310, 0.        , 0.        ,
         0.        , 0.        , 0.        , 0.04857296],
        [0.30393055, 0.        , 0.        , 0.        , 0.94133604, 0.        ,
         0.        , 0.        , 0.44766852, 1.71645379],
        [0.        , 0.44922280, 0.        , 0.59144729, 0.        , 1.05288684,
         0.        , 0.        , 0.        , 0.30007046],
        [1.49371696, 0.54554445, 0.40354243, 0.        , 1.32116580, 0.        ,
         0.        , 0.        , 0.        , 0.        ]])
➜  tmp_setuptools git:(setuptools80) ✗ l
total 36K
drwxr-xr-x  4 1000 1000 4.0K Nov  3 06:45 .
drwxr-xr-x 11 1000 1000 4.0K Oct 22 06:08 ..
drwxr-xr-x  3 root root 4.0K Nov  3 06:45 build
drwxr-xr-x  2 root root 4.0K Nov  3 06:45 custom_setup_ops.egg-info
-rw-rw-r--  1 1000 1000  464 Oct 21 10:07 readme.md
-rw-rw-r--  1 1000 1000 2.5K Oct 21 14:04 relu_cpu.cc
-rw-r--r--  1 root root  166 Oct 30 07:40 setup.py
-rw-rw-r--  1 1000 1000  166 Oct 21 14:06 setup_cpu.py
-rw-rw-r--  1 1000 1000  173 Oct 23 07:27 test.py
➜  tmp_setuptools git:(setuptools80) ✗ rm -rf build custom_setup_ops.egg-info
➜  tmp_setuptools git:(setuptools80) ✗ pip uninstall custom_setup_ops
Found existing installation: custom_setup_ops 0.0.0
Uninstalling custom_setup_ops-0.0.0:
  Would remove:
    /usr/local/lib/python3.9/dist-packages/custom_setup_ops-0.0.0.dist-info/*
    /usr/local/lib/python3.9/dist-packages/custom_setup_ops/*
    /usr/local/lib/python3.9/dist-packages/version.txt
Proceed (Y/n)? y
  Successfully uninstalled custom_setup_ops-0.0.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
➜  tmp_setuptools git:(setuptools80) ✗ ls -al /usr/local/lib/python3.9/dist-packages | grep custom
➜  tmp_setuptools git:(setuptools80) ✗ python setup.py install
[2025-11-03 06:45:49,342] [    INFO] dist.py:1018 - running install
/usr/local/lib/python3.9/dist-packages/setuptools/_distutils/cmd.py:90: SetuptoolsDeprecationWarning: setup.py install is deprecated.
!!

        ********************************************************************************
        Please avoid running ``setup.py`` directly.
        Instead, use pypa/build, pypa/installer or other
        standards-based tools.

        This deprecation is overdue, please update your project and remove deprecated
        calls to avoid build errors in the future.

        See https://blog.ganssle.io/articles/2021/10/setup-py-deprecated.html for details.
        ********************************************************************************

!!
  self.initialize_options()
[2025-11-03 06:45:49,345] [    INFO] dist.py:1018 - running build
[2025-11-03 06:45:49,345] [    INFO] dist.py:1018 - running build_ext
Compiling user custom op, it will cost a few seconds.....
[2025-11-03 06:45:49,365] [    INFO] build_ext.py:538 - building 'custom_setup_ops' extension
[2025-11-03 06:45:49,365] [    INFO] dir_util.py:58 - creating /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-cpython-39/build/custom_setup_ops/temp.linux-x86_64-cpython-39
[2025-11-03 06:45:49,365] [    INFO] spawn.py:77 - x86_64-linux-gnu-g++ -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -fPIC -I/usr/local/lib/python3.9/dist-packages/paddle/include -I/usr/local/lib/python3.9/dist-packages/paddle/include/third_party -I/usr/local/lib/python3.9/dist-packages/paddle/include/paddle/phi/api/include/compat -I/usr/local/lib/python3.9/dist-packages/paddle/include/paddle/phi/api/include/compat/torch/csrc/api/include -I/usr/include/python3.9 -I/usr/include/python3.9 -c /paddle/Paddle/build/tmp_setuptools/relu_cpu.cc -o /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-cpython-39/build/custom_setup_ops/temp.linux-x86_64-cpython-39/relu_cpu.o -w -DPADDLE_WITH_CUSTOM_KERNEL -DPADDLE_EXTENSION_NAME=custom_setup_ops -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++17
/paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-cpython-39/build/custom_setup_ops/temp.linux-x86_64-cpython-39/relu_cpu.o is compiled
[2025-11-03 06:45:54,256] [    INFO] spawn.py:77 - x86_64-linux-gnu-g++ -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -shared -Wl,-O1 -Wl,-Bsymbolic-functions /paddle/Paddle/build/tmp_setuptools/build/custom_setup_ops/lib.linux-x86_64-cpython-39/build/custom_setup_ops/temp.linux-x86_64-cpython-39/relu_cpu.o -L/usr/local/lib/python3.9/dist-packages/paddle/libs -L/usr/local/lib/python3.9/dist-packages/paddle/base -L/usr/lib/x86_64-linux-gnu -Wl,--enable-new-dtags,-rpath,/usr/local/lib/python3.9/dist-packages/paddle/libs -Wl,--enable-new-dtags,-rpath,/usr/local/lib/python3.9/dist-packages/paddle/base -o build/custom_setup_ops/lib.linux-x86_64-cpython-39/custom_setup_ops.so -l:libpaddle.so
Received len(custom_op) = 1, using custom operator
Removed: build/custom_setup_ops/lib.linux-x86_64-cpython-39/build/custom_setup_ops/temp.linux-x86_64-cpython-39/relu_cpu.o
[2025-11-03 06:45:54,678] [    INFO] dist.py:1018 - running install_lib
[2025-11-03 06:45:54,686] [    INFO] file_util.py:130 - copying build/custom_setup_ops/lib.linux-x86_64-cpython-39/version.txt -> /usr/local/lib/python3.9/dist-packages
[2025-11-03 06:45:54,686] [    INFO] file_util.py:130 - copying build/custom_setup_ops/lib.linux-x86_64-cpython-39/custom_setup_ops.py -> /usr/local/lib/python3.9/dist-packages
[2025-11-03 06:45:54,686] [    INFO] file_util.py:130 - copying build/custom_setup_ops/lib.linux-x86_64-cpython-39/custom_setup_ops.so -> /usr/local/lib/python3.9/dist-packages
[2025-11-03 06:45:54,689] [    INFO] util.py:485 - byte-compiling /usr/local/lib/python3.9/dist-packages/custom_setup_ops.py to custom_setup_ops.cpython-39.pyc
[2025-11-03 06:45:54,689] [    INFO] dist.py:1018 - running install_egg_info
[2025-11-03 06:45:54,707] [    INFO] dist.py:1018 - running egg_info
[2025-11-03 06:45:54,714] [    INFO] dir_util.py:58 - creating custom_setup_ops.egg-info
[2025-11-03 06:45:54,715] [    INFO] egg_info.py:651 - writing custom_setup_ops.egg-info/PKG-INFO
[2025-11-03 06:45:54,715] [    INFO] egg_info.py:279 - writing dependency_links to custom_setup_ops.egg-info/dependency_links.txt
[2025-11-03 06:45:54,715] [    INFO] egg_info.py:279 - writing top-level names to custom_setup_ops.egg-info/top_level.txt
[2025-11-03 06:45:54,715] [    INFO] util.py:332 - writing manifest file 'custom_setup_ops.egg-info/SOURCES.txt'
[2025-11-03 06:45:54,722] [    INFO] sdist.py:203 - reading manifest file 'custom_setup_ops.egg-info/SOURCES.txt'
[2025-11-03 06:45:54,723] [    INFO] util.py:332 - writing manifest file 'custom_setup_ops.egg-info/SOURCES.txt'
[2025-11-03 06:45:54,723] [    INFO] util.py:332 - Copying custom_setup_ops.egg-info to /usr/local/lib/python3.9/dist-packages/custom_setup_ops-0.0.0-py3.9.egg-info
[2025-11-03 06:45:54,724] [    INFO] dist.py:1018 - running install_scripts
➜  tmp_setuptools git:(setuptools80) ✗ python test.py
Tensor(shape=[4, 10], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[0.29788449, 1.78881752, 0.        , 0.92925411, 1.18475580, 0.        ,
         0.66181940, 0.        , 0.        , 0.        ],
        [0.82853365, 0.50783914, 1.11154175, 0.        , 0.        , 1.60252881,
         1.89361107, 0.        , 1.11063087, 1.98141348],
        [1.42123890, 0.        , 0.        , 0.        , 0.        , 0.74431223,
         0.        , 0.        , 0.92516363, 0.        ],
        [1.34734118, 0.        , 0.31587631, 0.00774950, 0.34321636, 0.04578846,
         0.        , 1.72909606, 0.        , 0.        ]])
➜  tmp_setuptools git:(setuptools80) ✗ l
total 36K
drwxr-xr-x  4 1000 1000 4.0K Nov  3 06:45 .
drwxr-xr-x 11 1000 1000 4.0K Oct 22 06:08 ..
drwxr-xr-x  3 root root 4.0K Nov  3 06:45 build
drwxr-xr-x  2 root root 4.0K Nov  3 06:45 custom_setup_ops.egg-info
-rw-rw-r--  1 1000 1000  464 Oct 21 10:07 readme.md
-rw-rw-r--  1 1000 1000 2.5K Oct 21 14:04 relu_cpu.cc
-rw-r--r--  1 root root  166 Oct 30 07:40 setup.py
-rw-rw-r--  1 1000 1000  166 Oct 21 14:06 setup_cpu.py
-rw-rw-r--  1 1000 1000  173 Oct 23 07:27 test.py
➜  tmp_setuptools git:(setuptools80) ✗ rm -rf build custom_setup_ops.egg-info
➜  tmp_setuptools git:(setuptools80) ✗ pip uninstall custom_setup_ops
Found existing installation: custom_setup_ops 0.0.0
Uninstalling custom_setup_ops-0.0.0:
  Would remove:
    /usr/local/lib/python3.9/dist-packages/custom_setup_ops
    /usr/local/lib/python3.9/dist-packages/custom_setup_ops-0.0.0-py3.9.egg-info
Proceed (Y/n)? y
  Successfully uninstalled custom_setup_ops-0.0.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
➜  tmp_setuptools git:(setuptools80) ✗ ls -al /usr/local/lib/python3.9/dist-packages | grep custom
➜  tmp_setuptools git:(setuptools80) ✗ l
total 28K
drwxr-xr-x  2 1000 1000 4.0K Nov  3 06:46 .
drwxr-xr-x 11 1000 1000 4.0K Oct 22 06:08 ..
-rw-rw-r--  1 1000 1000  464 Oct 21 10:07 readme.md
-rw-rw-r--  1 1000 1000 2.5K Oct 21 14:04 relu_cpu.cc
-rw-r--r--  1 root root  166 Oct 30 07:40 setup.py
-rw-rw-r--  1 1000 1000  166 Oct 21 14:06 setup_cpu.py
-rw-rw-r--  1 1000 1000  173 Oct 23 07:27 test.py


```

在 `BuildExtension` 类中添加了 `_generate_python_api_file` 和 `custom_write_stub` 方法之后，实际上 `extension_utils.py` 中的 `bootstrap_context` 已经不需要了，因为每次 `BuildExtension` 都会执行构建，但是建议先保留 `bootstrap_context` 以防其他兼容性问题 (比如外部用户使用等)：

``` python

@contextmanager
def bootstrap_context():
    """
    Context to manage how to write `__bootstrap__` code in .egg
    """
    origin_write_stub = bdist_egg.write_stub
    bdist_egg.write_stub = custom_write_stub # 这里注释掉后，不影响安装
    yield

    bdist_egg.write_stub = origin_write_stub

```

改动之后，测试用例中需要修改 `len(custom_egg_path) == 2` 。

总结：

- 增加兼容 `pip install .` 的方式安装
- 增加兼容 `pip install .` 的方式安装之后，实际上已经解决了 setuptools 版本的兼容性问题，不需要针对 setuptools 的版本再进行判断
- 需要修改测试用例


# 六、测试和验收的考量

## 测试用例

1. **基础功能测试**：
   - 在 Setuptools 80+ 环境下编译和安装自定义算子
   - 验证生成的包结构正确
   - 验证 Python stub 文件正确生成
   - 验证共享库正确重命名

2. **兼容性测试**：
   - 在 Setuptools >= 80 环境下验证功能正常
   - 在 Setuptools < 80 环境下验证功能正常

3. **导入测试**：
   - 验证安装后能够正确导入自定义算子
   - 验证算子功能正常运行

4. **pip 集成测试**：
   - 验证 `pip list` 能够正确显示已安装的自定义算子
   - 验证 `pip show` 能够正确显示自定义算子信息
   - 验证 `pip uninstall` 能够正确卸载

## 验收标准

1. 所有现有测试用例通过
2. 在 Setuptools 80+ 环境下，自定义算子能够正常编译、安装和运行
3. `pip list` 能够正确显示已安装的自定义算子包
4. 不影响旧版本 Setuptools 的功能

# 七、可行性分析和排期规划

## 可行性分析

1. **技术可行性**：方案基于 setuptools 的标准扩展机制，技术上完全可行
2. **兼容性风险**：通过条件判断确保兼容 80.0- 的 Setuptools
3. **测试覆盖**：现有测试用例能够覆盖主要功能点

## 排期规划

- **第 1 周**：完成核心代码实现和单元测试
- **第 2 周**：代码审查和合并

# 八、影响面

自定义算子用户无需修改现有代码，透明升级

# 附件及参考资料

1. [Setuptools 80.0 Release Notes](https://setuptools.pypa.io/en/latest/history.html#v80-0-0)
2. [PEP 566 - Metadata for Python Software Packages 2.1](https://peps.python.org/pep-0566/)
3. [Hackathon 9th No.109 任务说明](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_9th/%E3%80%90Hackathon_9th%E3%80%91%E4%B8%AA%E4%BA%BA%E6%8C%91%E6%88%98%E8%B5%9B%E2%80%94%E6%A1%86%E6%9E%B6%E5%BC%80%E5%8F%91%E4%BB%BB%E5%8A%A1%E5%90%88%E9%9B%86.md#no109-基于-setuptools-80-版本自定义算子机制适配)
