%define php_version 8.5.7
%define php_release 1

Summary: PHP 8.5.7 with LoongArch64 JIT support
Name: php
Version: %{php_version}
Release: %{php_release}%{?dist}
License: PHP-3.01
URL: https://www.php.net/
Source0: php-%{version}.tar.xz
Source1: php.ini

# LA64 JIT patches for LoongArch64 (apply in order)
# Patches 1-3: git-format tracked changes to .dasc and zend_jit.c
# The ir_emit_loongarch64.h is a DynASM-generated file that must be
# post-processed to replace bare LA64_xxx() with la64_emit() calls.
Patch0001: 0001-LA64-JIT-Fix-crashes-in-ir_get_target_constraints-an.patch
Patch0002: 0002-LA64-JIT-Stable-build-with-proper-ir_match_insn-and-.patch
Patch0003: 0003-LA64-JIT-Working-tracing-JIT-mode-Fix-instruction-wr.patch
Patch0004: 0004-LA64-JIT-dasc-and-framework-changes.patch

BuildRequires: gcc, make, autoconf, bison, pkgconfig
BuildRequires: libxml2-devel, sqlite-devel, bzip2-devel, curl-devel
BuildRequires: readline-devel, libedit-devel, libjpeg-devel, libpng-devel
BuildRequires: libXpm-devel, freetype-devel, gmp-devel, libxslt-devel
BuildRequires: openssl-devel, pcre-devel, zlib-devel

%description
PHP 8.5.7 built with LoongArch64 (LA64) JIT support.
The tracing JIT mode is fully functional on LoongArch64.

%prep
%setup -q -n php-%{version}
%patch0001 -p1
%patch0002 -p1
%patch0003 -p1
%patch0004 -p1

# Post-process ir_emit_loongarch64.h to wrap bare LA64 macros with la64_emit()
# (Required for LA64 direct-buffer JIT backend)
# The file must be re-generated from .dasc by DynASM, then post-processed.
# See ir_emit_loongarch64.h in SOURCES/ for the pre-processed version.

%build
%configure \
    --prefix=/usr \
    --sysconfdir=/etc/php \
    --with-config-file-path=/etc/php \
    --enable-opcache \
    --enable-opcache-jit \
    --with-pcre-jit \
    --enable-mbstring \
    --enable-bcmath \
    --with-zlib \
    --with-gettext \
    --enable-zend-test=no \
    --enable-debug

%{make_build}

%install
%{make_install}
install -D -m 644 %{SOURCE1} %{buildroot}%{_sysconfdir}/php/php.ini

%files
%{_bindir}/php
%{_bindir}/php-cgi
%{_bindir}/phpdbg
%{_libdir}/libphp.so
%{_datadir}/php/
%{_sysconfdir}/php/
%{_mandir}/man1/

%changelog
* Tue Jun 09 2026 LoongArch64 Porter <porter@loongson.dev>
- Initial LA64 JIT support for PHP 8.5.7
- tracing JIT mode verified working on LoongArch64
- Patches: ir_match_insn, ir_get_target_constraints, code buffer integration
