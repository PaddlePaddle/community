%global pysrcver 2.7
%global pyver    2.7.18
%global pyname   python2

# Python 2.7 uses #!/usr/bin/env python shebangs (standard practice)
# Anolis brp-mangle-shebangs rejects this; disable it for this package
%global __brp_mangle_shebangs %{nil}
# Python 2.7 not available as system python during build
%global __brp_python_bytecompile %{nil}

Summary: Python 2.7.18 runtime for LoongArch64
Name:    %{pyname}
Version: 2.7.18
Release: 1%{?dist}
License: Python-2.0
URL:     https://www.python.org/
Source0: Python-%{pyver}.tar.xz

# Patch: updated config.guess/config.sub to recognize loongarch64
Patch0:  python2.7-loongarch64-config-guess.patch
# Patch: fix OpenSSL 3.x version detection for _hashlib
Patch1:  python2.7-openssl3-hashlib.patch

BuildRequires: gcc, make, glibc-devel
BuildRequires: zlib-devel, bzip2-devel, xz-devel
BuildRequires: openssl-devel, libffi-devel
BuildRequires: readline-devel, ncurses-devel
BuildRequires: sqlite-devel, gdbm-devel
BuildRequires: tk-devel, tcl-devel
BuildRequires: expat-devel
BuildRequires: pkgconfig

Requires:      %{name}-libs%{?_isa} = %{version}-%{release}
Requires(post): /sbin/ldconfig
Requires(postun): /sbin/ldconfig

%description
Python 2.7.18 is the last release of Python 2, ported to LoongArch64.
This build includes:
- Core interpreter and standard library
- ctypes (using system libffi 3.4.4)
- hashlib / ssl (using OpenSSL 3.x)
- sqlite3, readline, curses, Tkinter support

%package libs
Summary: Python 2.7 runtime libraries
Requires: %{name}%{?_isa} = %{version}-%{release}

%description libs
Python 2.7 shared runtime library (libpython2.7.so).
Required by all programs embedding or extending Python 2.7.

%package devel
Summary: Python 2.7 development headers and libraries
Requires: %{name}%{?_isa} = %{version}-%{release}
Requires: %{name}-libs%{?_isa} = %{version}-%{release}

%description devel
Python 2.7 development headers, static library, config, and pkg-config files.
Required for building C extensions for Python 2.7.

%prep
%setup -q -n Python-%{pyver}
%patch -P 0 -p1 -b .config-guess-orig
%patch -P 1 -p1 -b .openssl3-orig

%build
# Replace config.guess/sub with loongarch64-aware versions
cp /usr/lib/rpm/anolis/config.guess config.guess
cp /usr/lib/rpm/anolis/config.sub config.sub

%configure \
    --with-system-ffi \
    --enable-shared \
    --with-threads \
    --with-signal-module \
    --with-wctype-functions \
    --enable-ipv6 \
    --with-system-expat \
    --with-dbmliborder=gdbm:bdb \
    %{nil}

make %{?_smp_mflags}

%install
make install DESTDIR=%{buildroot} INSTALL="install -p"

# Remove unwanted files
rm -f %{buildroot}%{_libdir}/python%{pysrcver}/config/libpython*.a
rm -f %{buildroot}%{_libdir}/python%{pysrcver}/config/python.o

# Remove precompiled .pyc/.pyo (they get regenerated)
find %{buildroot} -name '*.pyc' -o -name '*.pyo' | xargs rm -f

%check
# Quick smoke test
PYTHONHOME=%{buildroot}%{_prefix} LD_LIBRARY_PATH=%{buildroot}%{_libdir} \
    %{buildroot}%{_bindir}/python -c "
import sys
assert sys.version.startswith('2.7.18')
print('Python 2.7 smoke test: PASS')
"

%post libs -p /sbin/ldconfig
%postun libs -p /sbin/ldconfig

%files
%doc README LICENSE Misc/NEWS
%{_bindir}/python
%{_bindir}/python2
%{_bindir}/python2.7
%{_bindir}/pydoc
%{_bindir}/idle
%{_bindir}/2to3
%{_bindir}/smtpd.py
/usr/lib/python%{pysrcver}/
%{_libdir}/python%{pysrcver}/
%{_mandir}/man1/python*.1*
%exclude /usr/lib/python%{pysrcver}/test/
%exclude %{_libdir}/python%{pysrcver}/test/
%exclude /usr/lib/python%{pysrcver}/*.pyc
%exclude /usr/lib/python%{pysrcver}/*.pyo
%exclude %{_libdir}/python%{pysrcver}/*.pyc
%exclude %{_libdir}/python%{pysrcver}/*.pyo

%files libs
%{_libdir}/libpython2.7.so.1.0
%{_libdir}/libpython2.7.so.1.0*

%files devel
%{_bindir}/python2.7-config
%{_bindir}/python2-config
%{_bindir}/python-config
%{_includedir}/python%{pysrcver}/
%{_libdir}/libpython2.7.so
%{_libdir}/pkgconfig/python.pc
%{_libdir}/pkgconfig/python2.pc
%{_libdir}/pkgconfig/python-2.7.pc
%{_libdir}/python%{pysrcver}/config/

%changelog
* Mon Jun 08 2026 Loong Dev <loongdev@local> - 2.7.18-1
- Initial port to LoongArch64
- Updated config.guess/config.sub for LA64 recognition
- Use system libffi instead of bundled (2013) version
- Fixed OpenSSL 3.x version detection for _hashlib module
