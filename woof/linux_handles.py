"""Linux process handles (pidfd) and sealable memory files (memfd), whatever the interpreter wraps.

THE BREAKAGE THIS PREVENTS
--------------------------
Job ownership signals through a pidfd and VTSM retention keeps its states in
sealed memfds, and both asked the interpreter for its wrappers
(``os.pidfd_open``, ``signal.pidfd_send_signal``, ``os.memfd_create``).  A
CPython built against old kernel and C-library headers omits those wrappers
although the kernel it runs on serves the calls: the conda-forge CPython 3.14
the project's own pixi environments install is one (built on a glibc 2.17 /
Linux 3.10 sysroot), and on it every remote job refused with "remote Python
needs Linux pidfd process handles" on an ordinary Linux 6.x host.

So each call goes to the interpreter's wrapper when it has one, and otherwise
straight to the kernel: ``pidfd_open`` and ``pidfd_send_signal`` through
``syscall(2)`` (their numbers, 434 and 424, are shared by every Linux
architecture since 5.1 except alpha), ``memfd_create`` through the C
library's own function (glibc 2.27+, musl).  The syscall route is used only
after a handle it opened on this very process has been read back from
``/proc/self/fdinfo`` as a pidfd naming this process, so a wrong number can
never pass for a process handle.

Deliberately stdlib-only and import-light: the remote job worker imports it.
"""

from __future__ import annotations

import os
import sys

#: Generic Linux syscall numbers (include/uapi/asm-generic/unistd.h, and the
#: same number in every per-architecture table since Linux 5.1).  Alpha alone
#: offsets new syscalls by 110 and is not served by the fallback.
_SYS_PIDFD_SEND_SIGNAL = 424
_SYS_PIDFD_OPEN = 434

_LIBC = None
_SYSCALL_PIDFD: bool | None = None


def _libc():
    global _LIBC
    if _LIBC is None:
        import ctypes

        _LIBC = ctypes.CDLL(None, use_errno=True)
    return _LIBC


def _raise_errno(what: str) -> None:
    import ctypes

    number = ctypes.get_errno()
    raise OSError(number, f"{what}: {os.strerror(number)}")


def _syscall_pidfd_open(pid: int, flags: int = 0) -> int:
    import ctypes

    fd = _libc().syscall(ctypes.c_long(_SYS_PIDFD_OPEN), ctypes.c_int(pid), ctypes.c_uint(flags))
    if fd < 0:
        _raise_errno("pidfd_open")
    return fd


def _syscall_pidfd_works() -> bool:
    """The kernel opened a pidfd on this process and fdinfo names it so."""
    global _SYSCALL_PIDFD
    if _SYSCALL_PIDFD is None:
        _SYSCALL_PIDFD = False
        if sys.platform.startswith("linux") and not os.uname().machine.startswith("alpha"):
            try:
                fd = _syscall_pidfd_open(os.getpid())
            except (OSError, AttributeError):
                return False
            try:
                with open(f"/proc/self/fdinfo/{fd}", encoding="ascii") as stream:
                    held = [line.split()[1] for line in stream if line.startswith("Pid:")]
                _SYSCALL_PIDFD = held == [str(os.getpid())]
            except (OSError, IndexError):
                pass
            finally:
                os.close(fd)
    return _SYSCALL_PIDFD


def has_pidfd() -> bool:
    """This interpreter can open a pidfd and signal through it."""
    import signal

    if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
        return True
    return _syscall_pidfd_works()


def pidfd_open(pid: int, flags: int = 0) -> int:
    """``os.pidfd_open``; ``ProcessLookupError`` when ``pid`` is gone."""
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid, flags)
    if not _syscall_pidfd_works():
        raise OSError("this platform has no Linux pidfd process handles")
    return _syscall_pidfd_open(pid, flags)


def pidfd_send_signal(fd: int, sig: int, siginfo=None, flags: int = 0) -> None:
    """``signal.pidfd_send_signal``; ``ProcessLookupError`` when the process has ended."""
    import signal

    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        native(fd, sig, siginfo, flags)
        return
    if siginfo is not None:
        raise TypeError("siginfo is not supported through the syscall fallback")
    if not _syscall_pidfd_works():
        raise OSError("this platform has no Linux pidfd process handles")
    import ctypes

    result = _libc().syscall(ctypes.c_long(_SYS_PIDFD_SEND_SIGNAL), ctypes.c_int(fd),
                             ctypes.c_int(int(sig)), ctypes.c_void_p(None), ctypes.c_uint(flags))
    if result != 0:
        _raise_errno("pidfd_send_signal")


def has_memfd() -> bool:
    """This interpreter can create a memfd."""
    if hasattr(os, "memfd_create"):
        return True
    if not sys.platform.startswith("linux"):
        return False
    try:
        return hasattr(_libc(), "memfd_create")
    except OSError:
        return False


def memfd_create(name: str, flags: int) -> int:
    """``os.memfd_create``, or the C library's own when the interpreter omits it."""
    native = getattr(os, "memfd_create", None)
    if native is not None:
        return native(name, flags)
    if not has_memfd():
        raise OSError("this platform has no Linux memfd_create")
    import ctypes

    function = _libc().memfd_create
    function.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    fd = function(os.fsencode(name), flags)
    if fd < 0:
        _raise_errno("memfd_create")
    return fd
