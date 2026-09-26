#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

#define MEM_WORDS 2048
#define MSIGN  0400000000000ULL
#define PMASK  0377777777777ULL
#define WMASK  0777777777777ULL
#define ONEBIT 0200000000000ULL
#define AMSIGN 02000000000000ULL
#define AMMASK 01777777777777ULL
#define AQMASK 0777777777777ULL
#define APSIGN 0400000000000ULL
#define AMASK  07777U
#define LMASK  0777777000000ULL
#define RMASK  0777777ULL

typedef struct {
    uint64_t M[MEM_WORDS];
    uint64_t AC, MQ;
    uint16_t IC;
    int overflow;
    int halted;
    uint64_t steps;
} CPU;

static double now_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER ctr;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&ctr);
    return (double)ctr.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
#endif
}

static int file_exists(const char *path) {
    if (!path || !*path) return 0;
#ifdef _WIN32
    DWORD a = GetFileAttributesA(path);
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
#endif
}

static int load_oct(CPU *c, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open oct file: %s\n", path); return 0; }
    memset(c->M, 0, sizeof(c->M));
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p==' ' || *p=='\t' || *p=='\r' || *p=='\n') ++p;
        if (!*p || *p=='#' || *p==';') continue;
        char *end = NULL;
        unsigned long a = strtoul(p, &end, 8);
        if (end==p) continue;
        size_t wi = (size_t)a;
        p = end;
        while (*p) {
            while (*p==' ' || *p=='\t') ++p;
            if (!*p || *p=='\r' || *p=='\n' || *p=='#' || *p==';') break;
            errno = 0;
            unsigned long long v = strtoull(p, &end, 8);
            if (end==p || errno) break;
            if (wi < MEM_WORDS) c->M[wi] = ((uint64_t)v) & WMASK;
            ++wi;
            p = end;
        }
    }
    fclose(f);
    return 1;
}

static inline uint64_t readp(CPU *c, unsigned wi) {
    return wi < MEM_WORDS ? c->M[wi] : 0ULL;
}
static inline void writep(CPU *c, unsigned wi, uint64_t v) {
    if (wi < MEM_WORDS) c->M[wi] = v & WMASK;
}

static inline int64_t sm_to_i(uint64_t v, uint64_t signbit) {
    uint64_t mag = v & (signbit - 1ULL);
    return (v & signbit) ? -(int64_t)mag : (int64_t)mag;
}

static int cpu_step(CPU *c) {
    if (c->halted) return 0;
    uint64_t srword = readp(c, c->IC >> 1);
    uint64_t temp = (c->IC & 1) ? srword : (srword >> 18);
    temp &= RMASK;
    c->IC = (uint16_t)((c->IC + 1) & AMASK);
    c->steps++;

    uint16_t opcode = (uint16_t)((temp >> 12) & 077);
    uint16_t ma = (uint16_t)(temp & AMASK);
    uint64_t ibr = readp(c, ma >> 1);
    uint64_t sr = ibr;
    if ((opcode & 040) == 0) {
        if (ma & 1) sr <<= 18;
        sr &= LMASK;
    }
    unsigned base = opcode & 037;

    switch (base) {
        case 0:
            c->halted = 1; c->IC = ma; return 0;
        case 8: return 0;
        case 1: c->IC = ma; return 0;
        case 4: if ((c->AC & AMMASK) == 0) c->IC = ma; return 0;
        case 2: if (c->overflow) c->IC = ma; c->overflow = 0; return 0;
        case 3: if ((c->AC & AMSIGN) == 0) c->IC = ma; return 0;
        case 10: c->AC = ((sr & MSIGN) << 2) | (sr & PMASK); return 0;
        case 6: c->AC = (((sr & MSIGN) ^ MSIGN) << 2) | (sr & PMASK); return 0;
        case 15: c->MQ = sr & WMASK; return 0;
        case 14:
        case 12:
        case 13: {
            uint64_t nv;
            if (base == 14) nv = c->MQ;
            else if (base == 12) {
                nv = c->AC & PMASK;
                if (c->AC & AMSIGN) nv |= MSIGN;
            } else {
                if (opcode & 040) {
                    uint64_t t = c->AC & PMASK;
                    if (c->AC & AMSIGN) t |= MSIGN;
                    nv = sr & t;
                } else {
                    nv = sr & ~(((uint64_t)AMASK) << 18);
                    nv |= c->AC & (((uint64_t)AMASK) << 18);
                }
            }
            if ((opcode & 040) == 0) {
                if (ma & 1) { ibr &= LMASK; nv >>= 18; }
                else { ibr &= RMASK; }
                nv |= ibr;
            }
            writep(c, ma >> 1, nv);
            return 0;
        }
        case 5: case 7: case 9: case 11: {
            uint64_t x = sr;
            if (base == 5) x ^= MSIGN;
            if (base == 7) x |= MSIGN;
            if (base == 11) x &= PMASK;
            int64_t acv = sm_to_i(c->AC, AMSIGN);
            int64_t sv = (x & MSIGN) ? -(int64_t)(x & PMASK) : (int64_t)(x & PMASK);
            int64_t z = acv + sv;
            uint64_t az = (uint64_t)(z < 0 ? -z : z);
            if (az > AMMASK) {
                c->overflow = 1;
                az &= AMMASK;
                z = z < 0 ? -(int64_t)az : (int64_t)az;
            }
            c->AC = (z < 0 ? AMSIGN : 0ULL) | (uint64_t)(z < 0 ? -z : z);
            return 0;
        }
        case 20: {
            unsigned cnt = ma & 0377; int s = !!(c->MQ & MSIGN); c->AC &= AQMASK;
            while (cnt--) {
                c->MQ <<= 1; c->AC <<= 1;
                if (c->MQ & MSIGN) c->AC |= 1;
                if (c->AC & APSIGN) c->overflow = 1;
            }
            c->AC &= AMMASK; c->MQ &= PMASK;
            if (s) { c->AC |= AMSIGN; c->MQ |= MSIGN; }
            return 0;
        }
        case 21: {
            unsigned cnt = ma & 0377; int s = !!(c->AC & AMSIGN); c->AC &= AMMASK; c->MQ &= PMASK;
            while (cnt--) {
                if (c->AC & 1) c->MQ |= MSIGN;
                c->MQ >>= 1; c->AC >>= 1;
            }
            c->AC &= AMMASK;
            if (s) { c->AC |= AMSIGN; c->MQ |= MSIGN; }
            return 0;
        }
        case 22: {
            unsigned cnt = ma & 0377; int s = !!(c->AC & AMSIGN); c->AC &= AQMASK;
            while (cnt--) { c->AC <<= 1; if (c->AC & APSIGN) c->overflow = 1; }
            c->AC &= AMMASK;
            if (s) c->AC |= AMSIGN;
            return 0;
        }
        case 23: {
            unsigned cnt = ma & 0377; int s = !!(c->AC & AMSIGN); c->AC &= AMMASK; c->AC >>= cnt;
            if (s) c->AC |= AMSIGN;
            return 0;
        }
        default:
            fprintf(stderr, "unsupported opcode %o at %o\n", opcode, (unsigned)((c->IC - 1) & AMASK));
            return 2;
    }
}

static int save_snapshot(CPU *c, const char *path) {
    if (!path || !*path) return 1;
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write snapshot: %s\n", path); return 0; }
    if (fwrite(c->M, sizeof(uint64_t), MEM_WORDS, f) != MEM_WORDS) { fclose(f); return 0; }
    fclose(f); return 1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ibm701_core OCT START_OCT [INSTR_LIMIT] [TIME_MS] [SNAPSHOT] [STOP_FILE]\n");
        return 64;
    }
    CPU c; memset(&c, 0, sizeof(c));
    if (!load_oct(&c, argv[1])) return 65;
    c.IC = (uint16_t)(strtoul(argv[2], NULL, 8) & AMASK);
    uint64_t limit = argc > 3 ? strtoull(argv[3], NULL, 10) : 2000000000ULL;
    double time_ms = argc > 4 ? strtod(argv[4], NULL) : 0.0;
    const char *snap = argc > 5 ? argv[5] : NULL;
    const char *stopfile = argc > 6 ? argv[6] : NULL;
    double deadline = time_ms > 0 ? now_ms() + time_ms : 0.0;
    int status = 0;
    const unsigned CHUNK = 8192;
    while (!c.halted && c.steps < limit) {
        unsigned todo = CHUNK;
        uint64_t remain = limit - c.steps;
        if (remain < todo) todo = (unsigned)remain;
        for (unsigned i=0; i<todo && !c.halted; ++i) {
            int e = cpu_step(&c);
            if (e) { status = 2; goto done; }
        }
        if (stopfile && file_exists(stopfile)) { status = 4; break; }
        if (deadline > 0 && now_ms() >= deadline) { status = 3; break; }
    }
    if (c.halted) status = 0;
    else if (status == 0 && c.steps >= limit) status = 1;

done:
    if (!save_snapshot(&c, snap)) return 66;
    printf("IBM701_CORE_RESULT status=%d halted=%d steps=%llu IC=%04o AC=%013llo MQ=%012llo overflow=%d\n",
           status, c.halted, (unsigned long long)c.steps, (unsigned)c.IC,
           (unsigned long long)c.AC, (unsigned long long)c.MQ, c.overflow);
    fflush(stdout);
    return status == 2 ? 67 : 0;
}
