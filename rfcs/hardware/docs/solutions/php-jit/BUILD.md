# PHP 8.5.7 LoongArch64 JIT — 完整构建步骤

## 环境要求

| 组件 | 版本要求 |
|------|----------|
| GCC | ≥ 13.0 (LA64 JIT 需要新版本) |
| Node.js | ≥ 20 (用于 DynASM 预处理) |
| make, autoconf, bison, pkgconfig | 标准构建工具 |
| libxml2-devel, sqlite-devel | PHP 构建依赖 |

## 单架构构建（LoongArch64 原生）

```bash
# 1. 解压源码
cd ~/src
tar xf php-8.5.7.tar.xz
cd php-8.5.7/php-8.5.7

# 2. 应用 LA64 JIT 补丁
patch -p1 < /path/to/0001-LA64-JIT-Fix-crashes-in-ir_get_target_constraints-an.patch
patch -p1 < /path/to/0002-LA64-JIT-Stable-build-with-proper-ir_match_insn-and-.patch
patch -p1 < /path/to/0003-LA64-JIT-Working-tracing-JIT-mode-Fix-instruction-wr.patch
patch -p1 < /path/to/0004-LA64-JIT-dasc-and-framework-changes.patch

# 3. 后处理 ir_emit_loongarch64.h（DynASM 生成后修复）
# 必须将 DynASM 输出的 bare LA64_xxx() 包裹为 la64_emit(ctx, ...)
cp /path/to/ir_emit_loongarch64.h ext/opcache/jit/ir/ir_emit_loongarch64.h
touch ext/opcache/jit/ir/ir_emit_loongarch64.h  # 防止被覆盖

# 4. 配置（JIT 默认启用）
./configure --prefix=/home/user/php-install \
            --enable-opcache \
            --with-pcre-jit \
            --enable-opcache-jit \
            --enable-debug

# 5. 构建
make -j$(nproc)

# 6. 测试
make test
```

## 双架构对比构建（可选）

### X86-64 参考构建
```bash
# 在 X86-64 机器上
./configure --prefix=/tmp/php-x86 --enable-opcache --with-pcre-jit --enable-opcache-jit
make -j$(nproc)
make test
```

### LoongArch64 构建
```bash
# 在 LA64 机器上
./configure --prefix=/tmp/php-la64 --enable-opcache --with-pcre-jit --enable-opcache-jit
make -j$(nproc)
make test
```

## RPM 构建

```bash
# 使用 rpmbuild
rpmbuild -ba ~/rpmbuild/SPECS/php.spec

# 或使用 mock 进行隔离构建
mock -r loongarch64-epel-9 --rebuild php-8.5.7-1.src.rpm
```

## 快速验证 JIT

创建测试文件 `test_jit.php`：
```php
<?php
function fib($n) {
    if ($n <= 1) return $n;
    return fib($n-1) + fib($n-2);
}
echo fib(10) . "\n";
```

测试追踪 JIT：
```bash
sapi/cli/php -d opcache.enable_cli=1 \
             -d opcache.jit=tracing \
             -d opcache.jit_buffer_size=64M \
             test_jit.php
# 期望输出: 55
```

## 已知问题

- **function JIT 模式**：编译通过但生成的机器码有指令编码错误（SIGILL）
- **tracing JIT 模式**：完全可用，核心测试套件 99.9% 通过
- **首次构建**：若配置报错缺少依赖，请安装对应 `-devel` 包重试
