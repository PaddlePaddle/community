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

Setuptools 80- 中，`paddle.utils.cpp_extension.setup` 函数的内部实现是通过 `write_stub` 机制生成 Python API 文件，该机制在 Setuptools 80+ 中不再自动触发。

因此：

- 保留旧版本 Setuptools 下的兼容性，在 Setuptools 80- 中，通过 `write_stub` 机制生成 Python API 文件
- 在 Setuptools 80+ 中，通过 `BuildExtension` 类生成 Python API 文件
- 在 Setuptools 80+ 中，通过 `InstallCommand` 类处理安装目录规范化
- 通过判断 `setuptools.__version__` 来判断是否是 Setuptools 80+ 版本，从而选择不同的实现方案

### 1. 扩展 BuildExtension 类

添加 `_generate_python_api_file` 方法，在编译完成后生成 Python API 文件：

```python
    def _generate_python_api_file(self) -> None:
        """
        Generate the top-level python api file (package stub) alongside the
        built shared library in build_lib. This replaces the legacy bdist_egg
        write_stub mechanism that is no longer triggered in setuptools >= 80.
        """
        try:
            outputs = self.get_outputs()
            if not outputs:
                return
            # We only support a single extension per setup()
            so_path = os.path.abspath(outputs[0])
            so_name = os.path.basename(so_path)
            build_dir = os.path.dirname(so_path)
            # The package name equals distribution name
            pkg_name = self.distribution.get_name()
            pyfile = os.path.join(build_dir, f"{pkg_name}.py")
            # Write stub; it will reference the _pd_ renamed resource at import time
            custom_write_stub(so_name, pyfile)
        except Exception as e:
            raise RuntimeError(f"Failed to generate python api file: {e}") from e

    def run(self):
        super().run()

        # Skip if using legacy bdist_egg mechanism (setuptools < 80)
        if not _is_legacy_setuptools():
            # Generate python API stub into build_lib for setuptools >= 80 installs
            self._generate_python_api_file()

        self._clean_intermediate_files()
```

### 2. 新增 InstallCommand 类

添加自定义的 `install` 命令类，处理以下任务：

1. **选择合适的安装目录**：确保安装到 sys.path 中的目录
2. **重命名共享库**：将 `{pkg}.so` 重命名为 `{pkg}_pd_.so`，避免与 Python stub 冲突
3. **规范化包布局**：将文件组织为单一的包目录结构

```python
class InstallCommand(install):
    """
    Extend install Command to:
      1) choose an install dir that is actually importable (on sys.path)
      2) ensure a single top-level entry for the package in site/dist-packages so
         legacy tests that expect a sole artifact (egg/package) keep working
      3) rename the compiled library to *_pd_.so to avoid shadowing the python stub
    """

    def finalize_options(self) -> None:
        super().finalize_options()
        # Build candidate site dirs: global + user + entries already on sys.path
        candidates = []
        candidates.extend(site.getsitepackages())
        usp = site.getusersitepackages()
        if usp:
            candidates.append(usp)
        for sp in sys.path:
            if isinstance(sp, str) and sp.endswith((
                'site-packages', 'dist-packages'
            )):
                candidates.append(sp)
        # De-dup while preserving order
        seen = set()
        ordered = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
        # Prefer a candidate that is actually on sys.path
        target = None
        for c in ordered:
            if c in sys.path and os.path.isdir(c):
                target = c
                break
        # Fallback: pick the first existing candidate
        if target is None:
            for c in ordered:
                if os.path.isdir(c):
                    target = c
                    break
        if target:
            self.install_lib = target
            self.install_purelib = target
            self.install_platlib = target

    def run(self, *args: Any, **kwargs: Any) -> None:
        super().run(*args, **kwargs)
        # First rename the shared library if present at top-level
        self._rename_shared_library()
        # Then canonicalize layout to a single top-level entry for this package
        self._single_entry_layout()

    def _rename_shared_library(self) -> None:
        install_dir = (
            getattr(self, 'install_lib', None)
            or getattr(self, 'install_purelib', None)
            or getattr(self, 'install_platlib', None)
        )
        if not install_dir or not os.path.isdir(install_dir):
            return
        pkg = self.distribution.get_name()
        suffix = (
            '.pyd' if IS_WINDOWS else ('.dylib' if OS_NAME.startswith('darwin') else '.so')
        )
        old = os.path.join(install_dir, f"{pkg}{suffix}")
        new = os.path.join(install_dir, f"{pkg}_pd_{suffix}")
        if os.path.exists(old):
            if os.path.exists(new):
                os.remove(new)
            os.rename(old, new)

    def _single_entry_layout(self) -> None:
        """
        Ensure only one top-level item in install_dir contains the package name by:
          - moving {pkg}.py -> {pkg}/__init__.py
          - moving {pkg}_pd_.so -> {pkg}/{pkg}_pd_.so
          - removing any {pkg}-*.egg-info left by setuptools install (only if dist-info exists)
        This keeps legacy tests that scan os.listdir(site_dir) happy.
        """
        install_dir = (
            getattr(self, 'install_lib', None)
            or getattr(self, 'install_purelib', None)
            or getattr(self, 'install_platlib', None)
        )
        if not install_dir or not os.path.isdir(install_dir):
            return
        pkg = self.distribution.get_name()
        # Check if dist-info exists
        has_dist_info = any(
            name.endswith('.dist-info') and name.startswith(pkg)
            for name in os.listdir(install_dir)
        )
        # Prepare paths
        pkg_dir = os.path.join(install_dir, pkg)
        py_src = os.path.join(install_dir, f"{pkg}.py")
        # Find compiled lib (renamed or not)
        suf_so = (
            '.pyd' if IS_WINDOWS else ('.dylib' if OS_NAME.startswith('darwin') else '.so')
        )
        so_candidates = [
            os.path.join(install_dir, f"{pkg}_pd_{suf_so}"),
            os.path.join(install_dir, f"{pkg}{suf_so}"),
        ]
        so_src = next((p for p in so_candidates if os.path.exists(p)), None)
        # Create package dir
        if not os.path.isdir(pkg_dir):
            os.makedirs(pkg_dir, exist_ok=True)
        # Move python stub to package/__init__.py if exists
        if os.path.exists(py_src):
            py_dst = os.path.join(pkg_dir, "__init__.py")
            if os.path.exists(py_dst):
                os.remove(py_dst)
            os.replace(py_src, py_dst)
        # Move shared lib into the package dir if exists
        if so_src and os.path.exists(so_src):
            so_dst = os.path.join(pkg_dir, os.path.basename(so_src))
            if os.path.exists(so_dst):
                os.remove(so_dst)
            os.replace(so_src, so_dst)
        # Remove egg-info entries for this package only if dist-info exists
        if has_dist_info:
            for name in os.listdir(install_dir):
                if name.startswith(f"{pkg}-") and name.endswith(".egg-info"):
                    p = os.path.join(install_dir, name)
                    if os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=False)
                    else:
                        os.remove(p)
```

### 3. 更新测试用例

修改测试用例中的断言，通过 Setuptools 的版本判断生成的文件和目录的数量：

```python
        egg_counts = 1 if int(setuptools.__version__.split('.')[0]) < 80 else 2
        assert len(custom_egg_path) == egg_counts, (
            f"Matched egg number is {len(custom_egg_path)}."
        )
```

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
