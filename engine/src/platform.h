/* platform.h — the OS calls this engine actually uses, in one place.
 *
 * WHY THIS EXISTS. The brief for this engine was "runs on any OS". It did not:
 * the loader used pread/open/fstat, CPU detection used getauxval, and the server
 * used BSD sockets and sigaction. All POSIX. On Windows that meant WSL2 only,
 * which is not the same claim as "runs on Windows".
 *
 * The surface is deliberately tiny -- ten calls -- because every one of them is
 * a place where two operating systems can disagree. Kept as a header with static
 * inlines so there is no extra translation unit and no link-order surprise.
 *
 * WHAT IS NOT ABSTRACTED, on purpose: OpenMP (the compilers all have it),
 * aligned_alloc (C11), and the SIMD intrinsics (already guarded by ISA, and
 * MSVC accepts the same _mm_* names). Abstracting those would add indirection
 * without removing a real portability failure.
 */
#ifndef COLI_PLATFORM_H
#define COLI_PLATFORM_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>   /* memset, used by the signal and OVERLAPPED setup below */

#if defined(_WIN32)
  #define COLI_WINDOWS 1
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <io.h>
  typedef SOCKET coli_sock;
  #define COLI_INVALID_SOCK INVALID_SOCKET
  typedef int64_t coli_off;
#else
  #define COLI_POSIX 1
  #include <unistd.h>
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <arpa/inet.h>
  #include <signal.h>
  #include <errno.h>
  typedef int coli_sock;
  #define COLI_INVALID_SOCK (-1)
  typedef off_t coli_off;
#endif

/* ---------------------------------------------------------------- files ---
 * Positional reads only. The loader reads tensors from scattered offsets and
 * never uses the file position, so exposing seek+read would invite a
 * thread-safety bug that pread does not have. */

static inline int coli_open_ro(const char *path) {
#if defined(COLI_WINDOWS)
    return _open(path, _O_RDONLY | _O_BINARY);
#else
    return open(path, O_RDONLY);
#endif
}

static inline void coli_close(int fd) {
#if defined(COLI_WINDOWS)
    if (fd >= 0) _close(fd);
#else
    if (fd >= 0) close(fd);
#endif
}

/* Returns bytes read, or -1. On Windows there is no pread: ReadFile with an
 * OVERLAPPED offset is the positional equivalent and, unlike _lseek+_read, does
 * not move the shared file pointer. */
static inline int64_t coli_pread(int fd, void *buf, size_t n, int64_t off) {
#if defined(COLI_WINDOWS)
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    int64_t done = 0;
    while (done < (int64_t)n) {
        DWORD chunk = (DWORD)((n - (size_t)done) > 0x40000000u ? 0x40000000u : (n - (size_t)done));
        OVERLAPPED ov; memset(&ov, 0, sizeof ov);
        int64_t p = off + done;
        ov.Offset     = (DWORD)(p & 0xFFFFFFFFu);
        ov.OffsetHigh = (DWORD)((uint64_t)p >> 32);
        DWORD got = 0;
        if (!ReadFile(h, (char*)buf + done, chunk, &got, &ov)) {
            return (GetLastError() == ERROR_HANDLE_EOF) ? done : -1;
        }
        if (got == 0) break;                     /* EOF */
        done += got;
    }
    return done;
#else
    /* Loop: a short pread is legal and a single call can return less than n. */
    int64_t done = 0;
    while (done < (int64_t)n) {
        ssize_t r = pread(fd, (char*)buf + done, n - (size_t)done, (coli_off)(off + done));
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) break;
        done += r;
    }
    return done;
#endif
}

static inline int64_t coli_fsize(int fd) {
#if defined(COLI_WINDOWS)
    LARGE_INTEGER sz;
    HANDLE h = (HANDLE)_get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE || !GetFileSizeEx(h, &sz)) return -1;
    return (int64_t)sz.QuadPart;
#else
    struct stat st;
    if (fstat(fd, &st) != 0) return -1;
    return (int64_t)st.st_size;
#endif
}

/* ------------------------------------------------------------- sockets --- */

static inline int coli_net_init(void) {
#if defined(COLI_WINDOWS)
    WSADATA w; return WSAStartup(MAKEWORD(2,2), &w) == 0;
#else
    /* A write to a closed socket must not kill the process. Windows has no
     * SIGPIPE at all, which is why this is a no-op there rather than an
     * unimplemented gap. */
    signal(SIGPIPE, SIG_IGN);
    return 1;
#endif
}

static inline void coli_closesocket(coli_sock s) {
#if defined(COLI_WINDOWS)
    if (s != COLI_INVALID_SOCK) closesocket(s);
#else
    if (s >= 0) close(s);
#endif
}

/* Install a handler that does NOT restart interrupted syscalls.
 *
 * This is the bug that shipped: signal() installs RESTARTING handlers on Linux,
 * so accept() resumed instead of returning EINTR and the server ignored SIGTERM,
 * keeping its listening socket bound after `kill`. sigaction with sa_flags = 0
 * is the fix, and it has to be the default here so nobody reintroduces it.
 * Windows has no such distinction; its handler fires and the accept loop is
 * unblocked by closing the socket. */
static inline void coli_on_signal(void (*fn)(int)) {
#if defined(COLI_WINDOWS)
    signal(SIGINT, fn); signal(SIGTERM, fn);
#else
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_handler = fn; sa.sa_flags = 0;          /* NOT SA_RESTART */
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
#endif
}

static inline int coli_sock_would_retry(void) {
#if defined(COLI_WINDOWS)
    return WSAGetLastError() == WSAEINTR;
#else
    return errno == EINTR;
#endif
}

#endif /* COLI_PLATFORM_H */
