/* Per-step CUDA API census and GPU-busy timeline for WOOF profiling.
 *
 * A development tool, not part of the model: tools/launch_profile/
 * cupti_counter.py compiles it on demand against the toolkit's CUPTI and
 * drives it through ctypes.  CuPy 14 dlopens libcuda/libcudart, so an
 * LD_PRELOAD interposer never sees its calls; CUPTI's callback API does.
 *
 * What it records between reset() and read():
 *   - every driver and runtime API call, classified once per callback id
 *     (launch / memcpy H2D / D2H / D2D / memset / sync / alloc / free /
 *     other).  Driver calls made from INSIDE a runtime call are not counted
 *     a second time (a thread-local depth tracks the runtime call);
 *   - every kernel, memcpy and memset the device EXECUTED (activity API),
 *     as [start, end) intervals.  read() returns their union -- the time the
 *     device was busy with this process's work -- and the per-tag sum of
 *     kernel durations, where a tag is whatever set_tag() said when the
 *     launch was ISSUED (matched through the correlation id).
 *
 * Pinned: counts are per process, all threads; the busy union is over this
 * context only.  On a card shared with other contexts time-slicing inflates
 * kernel durations, which is a caveat on busy time, not on the counts.
 */
#include <cupti.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    C_TOTAL = 0, C_LAUNCH, C_GRAPH_LAUNCH, C_H2D, C_D2H, C_D2D, C_MEMCPY_OTHER,
    C_MEMSET, C_SYNC, C_SYNC_COPY, C_ALLOC, C_FREE, C_QUERY, C_OTHER,
    C_NCOUNTERS
};
#define NTAGS 16
#define CORR_RING (1u << 22)

static CUpti_SubscriberHandle g_sub;
static volatile long long g_count[C_NCOUNTERS];
static volatile long long g_tag_launch[NTAGS];
static unsigned char g_class_drv[CUPTI_DRIVER_TRACE_CBID_SIZE + 1];
static unsigned char g_class_rt[CUPTI_RUNTIME_TRACE_CBID_SIZE + 1];
static unsigned char g_class_drv_init[CUPTI_DRIVER_TRACE_CBID_SIZE + 1];
static unsigned char g_class_rt_init[CUPTI_RUNTIME_TRACE_CBID_SIZE + 1];
static unsigned char *g_corr_tag;
static volatile long long g_cbid_drv[CUPTI_DRIVER_TRACE_CBID_SIZE + 1];
static volatile long long g_cbid_rt[CUPTI_RUNTIME_TRACE_CBID_SIZE + 1];
static const char *g_name_drv[CUPTI_DRIVER_TRACE_CBID_SIZE + 1];
static const char *g_name_rt[CUPTI_RUNTIME_TRACE_CBID_SIZE + 1];
static volatile int g_tag;
static __thread int t_rt_depth;
static __thread uint64_t t_sync_enter;
/* Host time spent inside calls that can block on the device: explicit
 * synchronisations and EVERY copy (a cudaMemcpyAsync to pageable host
 * memory -- what CuPy's .get() issues -- returns only once the copy, and so
 * everything queued before it, has finished).  wall - sync_wait is the
 * host's own submission time. */
static volatile long long g_sync_wait_ns;

static int is_sync_class(int cls) {
    int base = cls & 0x0f;
    if (cls > 0x0f) return 1;                 /* a blocking copy */
    return base == C_SYNC || base == C_H2D || base == C_D2H || base == C_D2D
        || base == C_MEMCPY_OTHER;
}

/* executed-work intervals.  CUPTI delivers activity buffers on its own
 * worker thread, so everything below is guarded by g_spans_lock. */
static pthread_mutex_t g_spans_lock = PTHREAD_MUTEX_INITIALIZER;
typedef struct { uint64_t start, end; } span_t;
static span_t *g_spans;
static size_t g_nspans, g_cap;
static uint64_t g_tag_busy[NTAGS];
static long long g_kernels_executed;
static long long g_dropped;

static int has(const char *name, const char *needle) {
    return strstr(name, needle) != NULL;
}

static int classify(const char *name) {
    if (!name) return C_OTHER;
    if (has(name, "GraphLaunch")) return C_GRAPH_LAUNCH;
    if (has(name, "LaunchKernel") || has(name, "LaunchCooperative")
        || !strcmp(name, "cudaLaunch") || has(name, "cudaLaunch_"))
        return C_LAUNCH;
    if (has(name, "Synchronize")) return C_SYNC;
    if (has(name, "Query")) return C_QUERY;
    if (has(name, "Memset")) return C_MEMSET;
    if (has(name, "Memcpy")) {
        int async = has(name, "Async");
        int cls;
        if (has(name, "HtoD") || has(name, "ToDevice")) cls = C_H2D;
        else if (has(name, "DtoH") || has(name, "FromDevice")) cls = C_D2H;
        else if (has(name, "DtoD") || has(name, "Peer")) cls = C_D2D;
        else cls = C_MEMCPY_OTHER;
        /* A synchronous copy is a host synchronisation as well. */
        return async ? cls : (C_SYNC_COPY | (cls << 4));
    }
    if (has(name, "Free")) return C_FREE;
    if (has(name, "Alloc") || has(name, "cudaMalloc") || has(name, "cuMemCreate"))
        return C_ALLOC;
    return C_OTHER;
}

static void count_class(int cls) {
    if ((cls & 0x0f) == C_SYNC_COPY && cls > 0x0f) {
        __sync_fetch_and_add(&g_count[C_SYNC_COPY], 1);
        __sync_fetch_and_add(&g_count[cls >> 4], 1);
        return;
    }
    __sync_fetch_and_add(&g_count[cls], 1);
}

static void CUPTIAPI on_api(void *userdata, CUpti_CallbackDomain domain,
                            CUpti_CallbackId cbid, const void *data) {
    (void)userdata;
    const CUpti_CallbackData *cb = (const CUpti_CallbackData *)data;
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        if (cb->callbackSite == CUPTI_API_EXIT) {
            t_rt_depth--;
            if (cbid <= CUPTI_RUNTIME_TRACE_CBID_SIZE && g_class_rt_init[cbid]
                && is_sync_class(g_class_rt[cbid]) && t_sync_enter) {
                uint64_t now = 0;
                cuptiGetTimestamp(&now);
                __sync_fetch_and_add(&g_sync_wait_ns,
                                     (long long)(now - t_sync_enter));
                t_sync_enter = 0;
            }
            return;
        }
        t_rt_depth++;
        if (cbid > CUPTI_RUNTIME_TRACE_CBID_SIZE) return;
        if (!g_class_rt_init[cbid]) {
            g_class_rt[cbid] = (unsigned char)classify(cb->functionName);
            g_class_rt_init[cbid] = 1;
            g_name_rt[cbid] = cb->functionName;
        }
        __sync_fetch_and_add(&g_cbid_rt[cbid], 1);
        __sync_fetch_and_add(&g_count[C_TOTAL], 1);
        int cls = g_class_rt[cbid];
        count_class(cls);
        if (is_sync_class(cls)) cuptiGetTimestamp(&t_sync_enter);
        if (cls == C_LAUNCH || cls == C_GRAPH_LAUNCH) {
            int tag = g_tag;
            __sync_fetch_and_add(&g_tag_launch[tag], 1);
        }
        g_corr_tag[cb->correlationId & (CORR_RING - 1)] = (unsigned char)g_tag;
        return;
    }
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return;
    if (t_rt_depth > 0) return;           /* internal to a runtime call */
    if (cb->callbackSite != CUPTI_API_ENTER) {
        if (cbid <= CUPTI_DRIVER_TRACE_CBID_SIZE && g_class_drv_init[cbid]
            && is_sync_class(g_class_drv[cbid]) && t_sync_enter) {
            uint64_t now = 0;
            cuptiGetTimestamp(&now);
            __sync_fetch_and_add(&g_sync_wait_ns,
                                 (long long)(now - t_sync_enter));
            t_sync_enter = 0;
        }
        return;
    }
    if (cbid > CUPTI_DRIVER_TRACE_CBID_SIZE) return;
    if (!g_class_drv_init[cbid]) {
        g_class_drv[cbid] = (unsigned char)classify(cb->functionName);
        g_class_drv_init[cbid] = 1;
        g_name_drv[cbid] = cb->functionName;
    }
    __sync_fetch_and_add(&g_cbid_drv[cbid], 1);
    __sync_fetch_and_add(&g_count[C_TOTAL], 1);
    int cls = g_class_drv[cbid];
    count_class(cls);
    if (is_sync_class(cls)) cuptiGetTimestamp(&t_sync_enter);
    if (cls == C_LAUNCH || cls == C_GRAPH_LAUNCH) {
        int tag = g_tag;
        __sync_fetch_and_add(&g_tag_launch[tag], 1);
    }
    g_corr_tag[cb->correlationId & (CORR_RING - 1)] = (unsigned char)g_tag;
}

#define BUF_SIZE (8u * 1024u * 1024u)

static void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                                      size_t *max_records) {
    *buffer = (uint8_t *)aligned_alloc(8, BUF_SIZE);
    *size = BUF_SIZE;
    *max_records = 0;
}

static void push_span(uint64_t s, uint64_t e) {
    if (g_nspans == g_cap) {
        size_t cap = g_cap ? 2 * g_cap : 65536;
        span_t *grown = (span_t *)realloc(g_spans, cap * sizeof(span_t));
        if (!grown) { g_dropped++; return; }
        g_spans = grown; g_cap = cap;
    }
    g_spans[g_nspans].start = s;
    g_spans[g_nspans].end = e;
    g_nspans++;
}

static void CUPTIAPI buffer_completed(CUcontext ctx, uint32_t stream,
                                      uint8_t *buffer, size_t size,
                                      size_t valid) {
    (void)ctx; (void)stream; (void)size;
    CUpti_Activity *record = NULL;
    pthread_mutex_lock(&g_spans_lock);
    while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
        switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            CUpti_ActivityKernel12 *k = (CUpti_ActivityKernel12 *)record;
            push_span(k->start, k->end);
            g_kernels_executed++;
            unsigned tag = g_corr_tag[k->correlationId & (CORR_RING - 1)];
            if (tag < NTAGS) g_tag_busy[tag] += k->end - k->start;
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            CUpti_ActivityMemcpy6 *m = (CUpti_ActivityMemcpy6 *)record;
            push_span(m->start, m->end);
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            CUpti_ActivityMemset4 *m = (CUpti_ActivityMemset4 *)record;
            push_span(m->start, m->end);
            break;
        }
        default:
            break;
        }
    }
    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(ctx, stream, &dropped);
    g_dropped += (long long)dropped;
    pthread_mutex_unlock(&g_spans_lock);
    free(buffer);
}

static int cmp_span(const void *a, const void *b) {
    const span_t *x = (const span_t *)a, *y = (const span_t *)b;
    return (x->start > y->start) - (x->start < y->start);
}

/* ---- the ctypes surface ---------------------------------------------- */

int woofprof_start(int activity) {
    g_corr_tag = (unsigned char *)calloc(CORR_RING, 1);
    if (!g_corr_tag) return -1;
    if (cuptiSubscribe(&g_sub, (CUpti_CallbackFunc)on_api, NULL) != CUPTI_SUCCESS)
        return -2;
    if (cuptiEnableDomain(1, g_sub, CUPTI_CB_DOMAIN_DRIVER_API) != CUPTI_SUCCESS)
        return -3;
    if (cuptiEnableDomain(1, g_sub, CUPTI_CB_DOMAIN_RUNTIME_API) != CUPTI_SUCCESS)
        return -4;
    if (activity) {
        if (cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed)
            != CUPTI_SUCCESS) return -5;
        if (cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) != CUPTI_SUCCESS)
            return -6;
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    }
    return 0;
}

void woofprof_set_tag(int tag) {
    g_tag = (tag >= 0 && tag < NTAGS) ? tag : 0;
}

void woofprof_flush(void) { cuptiActivityFlushAll(1); }

void woofprof_reset(void) {
    cuptiActivityFlushAll(1);
    memset((void *)g_count, 0, sizeof(g_count));
    pthread_mutex_lock(&g_spans_lock);
    memset((void *)g_tag_launch, 0, sizeof(g_tag_launch));
    memset((void *)g_cbid_drv, 0, sizeof(g_cbid_drv));
    memset((void *)g_cbid_rt, 0, sizeof(g_cbid_rt));
    memset(g_tag_busy, 0, sizeof(g_tag_busy));
    g_nspans = 0;
    g_kernels_executed = 0;
    g_dropped = 0;
    g_sync_wait_ns = 0;
    pthread_mutex_unlock(&g_spans_lock);
}

/* counters[C_NCOUNTERS]; tag_launch[NTAGS]; tag_busy_ns[NTAGS];
 * extra[5] = {busy_union_ns, first_start_ns, last_end_ns, kernels_executed,
 *             sync_wait_ns};
 * returns records dropped. */
long long woofprof_read(long long *counters, long long *tag_launch,
                        long long *tag_busy_ns, long long *extra) {
    cuptiActivityFlushAll(1);
    pthread_mutex_lock(&g_spans_lock);
    for (int i = 0; i < C_NCOUNTERS; i++) counters[i] = g_count[i];
    for (int i = 0; i < NTAGS; i++) {
        tag_launch[i] = g_tag_launch[i];
        tag_busy_ns[i] = (long long)g_tag_busy[i];
    }
    uint64_t busy = 0, first = 0, last = 0;
    if (g_nspans) {
        qsort(g_spans, g_nspans, sizeof(span_t), cmp_span);
        uint64_t cs = g_spans[0].start, ce = g_spans[0].end;
        first = cs;
        last = ce;
        for (size_t i = 1; i < g_nspans; i++) {
            if (g_spans[i].start > ce) {
                busy += ce - cs;
                cs = g_spans[i].start;
                ce = g_spans[i].end;
            } else if (g_spans[i].end > ce) {
                ce = g_spans[i].end;
            }
            if (g_spans[i].end > last) last = g_spans[i].end;
        }
        busy += ce - cs;
    }
    extra[0] = (long long)busy;
    extra[1] = (long long)first;
    extra[2] = (long long)last;
    extra[3] = g_kernels_executed;
    extra[4] = g_sync_wait_ns;
    long long dropped = g_dropped;
    pthread_mutex_unlock(&g_spans_lock);
    return dropped;
}

/* Per-API call counts since reset: the i-th distinct API seen (driver ids
 * first, then runtime ids offset by 100000).  Returns the name or NULL. */
const char *woofprof_api(int i, long long *count) {
    int seen = 0;
    for (int d = 0; d <= CUPTI_DRIVER_TRACE_CBID_SIZE; d++) {
        if (!g_cbid_drv[d]) continue;
        if (seen++ == i) { *count = g_cbid_drv[d]; return g_name_drv[d]; }
    }
    for (int r = 0; r <= CUPTI_RUNTIME_TRACE_CBID_SIZE; r++) {
        if (!g_cbid_rt[r]) continue;
        if (seen++ == i) { *count = g_cbid_rt[r]; return g_name_rt[r]; }
    }
    return NULL;
}

int woofprof_ncounters(void) { return C_NCOUNTERS; }
int woofprof_ntags(void) { return NTAGS; }
