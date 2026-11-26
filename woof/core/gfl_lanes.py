"""Lane-interleave the GFL column arrays, the way CUDA lays out local memory.

The transcription (woof.core.gfl_873b414) gives every thread one contiguous
scratch arena, and the launcher handed it column-contiguous [ncol, levels]
fields, so a warp reading ``a[k]`` touched 32 addresses 91-266 KB apart (a
32-way scatter on every load and store). Measured on the captured WRF v4.8.0
CONUS 12 km columns this made the GFL call 5.3-6.0x today's cu_physics=3
kernel (GFDRV, gf.cu, whose workspace is interleaved: GFWS_LANES).

Here every float/integer/logical array pointer of the transcription becomes a
lane pointer whose element ``i`` lives ``i*GF_LANES`` words after element 0:
element ``e`` of lane ``t`` in a GF_LANES-column slab sits at
``slab + (e*GF_LANES + t)*4``, so a warp reading ``a[k]`` reads 32 consecutive
words. The level fields the launcher passes use the same placement
([slab][level][lane]). Addressing stays linear in the element index, so
``&a[e]`` handed to a callee and indexed there by ``j`` reaches element
``e+j`` exactly as before; GNU's byte displacements on descriptor data
(POINTER_PLUS_EXPR) become the same displacement in elements. The arithmetic,
its order and every index expression are unchanged: only placement moves.
Character arrays (ierrc, ierrcm, ierrcs) keep a small contiguous per-thread
region. The host witness compiles with ``GF_LANES 1``, the original layout.

The same transform as woof hex's GF lanes (woof/hex/mpasphys/gf_lanes.py,
commit 9adb7c0, GF 112 -> 27 ms/step), extended to GFL's descriptor byte
displacements, logical arrays and the deep-column spill, with warps taking
columns from a counter (a fixed column-per-slot round left its last round a
few blocks wide: 16.6 -> 14.2 ms on the same columns). It runs once at
kernel setup; nothing here runs in a step. The pinned transcription
(``_SCHEME_SOURCE``, transcription.json) is not edited.
"""
import re

#: Columns per slab, and threads per block (a slab is a block).
LANES = 128

#: Contiguous character bytes per thread: ierrc, ierrcm and ierrcs are one
#: 50-byte element each at its=ite=1, each 16-byte rounded (3 x 64 = 192),
#: and fscav's one word (16).
CHARACTER_BYTES = 256
#: The same per-thread contiguous region on Metal, which also holds every local whose address the
#: scheme passes on (:func:`metal_locals`: about 1.6 KB at most, plus the entry's 18 bounds).
METAL_CHARACTER_BYTES = 2048


def character_bytes(metal=False):
    """Contiguous bytes per lane for this backend's lowering (GF_CHARACTER_BYTES)."""
    return METAL_CHARACTER_BYTES if metal else CHARACTER_BYTES

#: Resident 128-thread blocks per SM the column kernel is compiled for
#: (``GFL_MIN_BLOCKS`` in the kernel defines below; a test keeps the two equal).
#: The admission prices the scratch with it before any kernel is compiled.
MIN_BLOCKS_PER_SM = 3

#: The most scratch the column launch holds at once.
SCRATCH_CAP_BYTES = 4 << 30
#: A lane's scratch share: bytes per level plus a fixed part. The deepest column of the
#: captured WRF v4.8.0 CONUS 12 km calls used 42,928 bytes at 50 storage levels; cut to 16,
#: 26, 36 and 50 levels they need 14,784, 24,112, 30,320 and 42,736 (about 820 per level),
#: against shares of 32,768, 43,008, 53,248 and 67,584. A column that needs more continues
#: in a spill entry of the launch's spill pool, so it stays exact and running.
SHARE_BYTES_PER_LEVEL = 1024
SHARE_BYTES_FIXED = 16384


def share_bytes(storage_levels, compute_levels):
    """One lane's scratch share for a column of this many storage and compute levels."""
    return SHARE_BYTES_PER_LEVEL * max(storage_levels, compute_levels) + SHARE_BYTES_FIXED


def scratch_blocks(slabs, per_slab, *, resident_per_sm, sms, free=None):
    """Blocks of the persistent column launch, each owning one slab of scratch.

    No more than the domain has slabs (a 150 x 150 domain is 176 slabs, not a
    whole card's 494), no more than the card keeps resident, and within
    SCRATCH_CAP_BYTES and a quarter of ``free`` when it is given. The block
    count changes no output word: columns are independent and each starts
    from a fresh workspace.
    """
    cap = SCRATCH_CAP_BYTES if free is None else min(SCRATCH_CAP_BYTES, int(free) // 4)
    return max(1, min(cap // per_slab, max(1, int(resident_per_sm)) * int(sms), max(1, int(slabs))))


def scratch_bytes(nz, columns, sms, *, resident_per_sm=MIN_BLOCKS_PER_SM, free=None, metal=False):
    """Device bytes the GFL column launch holds for ``columns`` columns of ``nz`` mass levels.

    The launcher (woof.core.gfl_873b414._column_scratch) allocates exactly
    this; the admission (woof.core.preflight.gf_column_workspace_bytes)
    prices it, with the spill pool (:func:`spill_pool_bytes`). Storage levels
    are the nz+1 faces, the widest field the adapter passes.
    """
    per_slab = (share_bytes(int(nz) + 1, int(nz)) + character_bytes(metal)) * LANES
    slabs = -(-max(int(columns), 1) // LANES)
    return scratch_blocks(slabs, per_slab, resident_per_sm=resident_per_sm, sms=sms, free=free) * per_slab


#: Spill shares per spill entry: a column deeper than its share continues in an entry this
#: many shares deep.
SPILL_SHARES = 2
#: The most the spill pool holds (upstream 1.1.7's device-heap limit for the same spill).
SPILL_POOL_BYTES = 256 << 20


def spill_pool_slabs(spill_bytes, slabs):
    """Slabs of the launch's spill pool: one per domain slab (an entry for every thread the
    launch can start), within SPILL_POOL_BYTES.

    A single slab larger than SPILL_POOL_BYTES (a column so deep that one entry per lane
    passes the cap) is refused by name rather than allocated over the cap.
    """
    per_slab = int(spill_bytes) * LANES
    if per_slab > SPILL_POOL_BYTES:
        raise ValueError(
            f"GFL spill slab of {per_slab} bytes ({int(spill_bytes)} per lane x {LANES} lanes) is larger "
            f"than the spill pool cap SPILL_POOL_BYTES={SPILL_POOL_BYTES}: the column is too deep for "
            "the GFL lane launch")
    return min(max(int(slabs), 1), SPILL_POOL_BYTES // per_slab)


def spill_pool_bytes(nz, columns):
    """Device bytes of the spill pool the GFL column launch holds for ``columns`` columns of
    ``nz`` mass levels (woof.core.gfl_873b414._spill_pool allocates exactly this)."""
    spill_bytes = SPILL_SHARES * share_bytes(int(nz) + 1, int(nz))
    return spill_pool_slabs(spill_bytes, -(-max(int(columns), 1) // LANES)) * spill_bytes * LANES

_DEFINES = r'''#ifndef GF_LANES
#ifdef GF_HOST_WITNESS
#define GF_LANES 1
#else
#define GF_LANES 128
#endif
#endif
// Resident 128-thread blocks per SM the column kernel is compiled for. Measured on the captured
// WRF v4.8.0 CONUS 12 km columns (126,776, step 3), quiet RTX PRO 6000, fast numerics, 2026-11-09:
// 1 block (255 registers) 14.21 ms, 2 (255) 14.21, 3 (168) 13.66, 4 (128) 13.97; GFDRV 11.8 ms.
#ifndef GFL_MIN_BLOCKS
#define GFL_MIN_BLOCKS 3
#endif
#define GF_CHARACTER_BYTES 256
// The launch counters (column, spill pool). 32-bit on Metal, which has no 64-bit atomic add (the
// launcher bounds both counters there); 64-bit on CUDA, whose kernel then compiles as it did with
// 64-bit counters (a 32-bit column counter cost a 64-bit-counter build ~40% on an A6000, 2026-11-23).
#ifdef __WOOF_METAL__
typedef unsigned int gf_counter;
#else
typedef unsigned long long gf_counter;
#endif
#ifdef GF_HOST_WITNESS
#define __forceinline__ inline
#define __launch_bounds__(...)
#endif
'''

_LANE_POINTER = r'''
// A column array pointer whose element i lives GF_LANES words after element
// i-1 (lane-interleaved placement). Arithmetic and dereference stay linear in
// the element index; a pointer to a scalar is only ever read at element 0.
// The word stride holds for 1-byte logicals too (flg), so no two lanes share
// a word.
template<class T> struct GFLanePtr {
  T* p;
  __device__ __forceinline__ GFLanePtr() = default;
  __device__ __forceinline__ GFLanePtr(T* q):p(q){}
  __device__ __forceinline__ explicit GFLanePtr(void* q):p((T*)q){}
  __device__ __forceinline__ GFLanePtr(decltype(nullptr)):p(nullptr){}
  // Fortran OPTIONAL arguments are tested against null.
  __device__ __forceinline__ bool operator==(decltype(nullptr)) const {return p==nullptr;}
  __device__ __forceinline__ bool operator!=(decltype(nullptr)) const {return p!=nullptr;}
  template<class I> __device__ __forceinline__ T& operator[](I i) const {return *(T*)((char*)p+(long long)i*(4*GF_LANES));}
  __device__ __forceinline__ T& operator*() const {return *p;}
  template<class I> __device__ __forceinline__ GFLanePtr operator+(I i) const {return GFLanePtr((T*)((char*)p+(long long)i*(4*GF_LANES)));}
  __device__ __forceinline__ explicit operator char*() const {return (char*)p;}
  __device__ __forceinline__ explicit operator void*() const {return (void*)p;}
};
typedef GFLanePtr<GFReal> GFP;
typedef GFLanePtr<int> GIP;
typedef GFLanePtr<bool> GBP;
template<class T> __device__ __forceinline__ char* gf_lane_raw(GFLanePtr<T> q){return (char*)q.p;}
__device__ __forceinline__ char* gf_lane_raw(void* q){return (char*)q;}
// GNU's POINTER_PLUS_EXPR on descriptor data: a byte displacement of a whole
// number of 4-byte elements (signed; GNU prints it through an unsigned cast)
// becomes the same number of elements in the lane placement.
template<class T,class P> __device__ __forceinline__ GFLanePtr<T> gf_lane_plus(P q,unsigned long long bytes){
  return GFLanePtr<T>((T*)(gf_lane_raw(q)+((long long)bytes/4)*(4*GF_LANES)));}
'''

_WORKSPACE_OLD = '''struct GFWorkspace {
  char* data;unsigned long long capacity,position,highwater;unsigned int diagnostics;GFReal row_dx_first;
  // Mutable SAVE, including the early return: WRF v4.8.0 phys/GFL 873b414 module_cu_gfl_deep.F90:2641-2705
  bool fct1d3_error;
  // A column whose need passes its share of the launch scratch continues in one device-heap block of spill_bytes
  // (gfl_column frees it): the share is sized to what columns use, the spill keeps a deeper column exact and running.
  char* spill;unsigned long long spill_bytes;
// Typed CUDA support for GFL support (WRF v4.8.0 phys/GFL 873b414)
  __device__ void* allocate(unsigned long long n){
    n=(n+15)&~15ULL;
    if(position+n>capacity){
#ifdef GF_HOST_WITNESS
      gf_trap();return nullptr;
#else
      if(position<capacity)position=capacity;
      if(spill==nullptr){spill=(char*)malloc(spill_bytes);if(spill==nullptr){gf_trap();return nullptr;}}
      if(position+n>capacity+spill_bytes){gf_trap();return nullptr;}
      void* result=spill+(position-capacity);position+=n;highwater=gf_max(highwater,position);return result;
#endif
    }
    void* result=data+position;position+=n;highwater=gf_max(highwater,position);return result;
  }
};
struct GFFrame {
 GFWorkspace &ws;unsigned long long saved;
 __device__ GFFrame(GFWorkspace &w):ws(w),saved(w.position){}
 __device__ ~GFFrame(){ws.position=saved;}
};'''

# Positions count this lane's bytes exactly as before (so capacity, highwater
# and the spill threshold keep their meaning); a lane byte offset b is the
# address data + b*GF_LANES, i.e. word b/4 of the lane-interleaved slab.
_WORKSPACE_NEW = '''struct GFWorkspace {
  char* data;unsigned long long capacity,position,highwater;unsigned int diagnostics;GFReal row_dx_first;
  // Mutable SAVE, including the early return: WRF v4.8.0 phys/GFL 873b414 module_cu_gfl_deep.F90:2641-2705
  bool fct1d3_error;
  // A column whose need passes its share continues in a spill entry of spill_bytes, lane-interleaved like the share:
  // entry e is word e%GF_LANES of slab e/GF_LANES of the launch's spill pool (host-allocated, spill_entries entries).
  // A column that spills takes a free entry (spill_entry, e+1, 0 = none) from the pool's bitmap (spill_map, bit e
  // set while entry e is held; 32-bit atomics, which Metal has) and the kernel returns it when the column ends.
  // Metal has no device heap, so the pool stands in for an in-kernel malloc on every backend: placement only, the
  // same words. The share is sized to what columns use, the spill keeps a deeper column exact and running.
  // The limit: at most spill_entries columns can be spilling AT THE SAME TIME; one more traps (gf_trap, a loud
  // failure, never a wrong word). Upstream 1.1.7 gave each spilling block a heap slab of 128 entries for the rest
  // of the block's run (256 MiB heap), so the pool admits every launch upstream did, and more.
  char* spill_pool;unsigned long long spill_bytes,spill_entries;unsigned int* spill_map;unsigned long long spill_entry;
  // Character arrays are read bytewise: contiguous per thread.
  char* bytes;unsigned long long bcapacity,bposition;unsigned int lane;
  __device__ void* allocate(unsigned long long n){
    n=(n+15)&~15ULL;
    if(position+n>capacity){
#ifdef GF_HOST_WITNESS
      gf_trap();return nullptr;
#else
      if(position<capacity)position=capacity;
      if(position+n>capacity+spill_bytes){gf_trap();return nullptr;}
      if(spill_entry==0ULL){
        // First free entry: scan the bitmap, claiming a clear bit with atomicOr (the old word says who won).
        const unsigned long long words=(spill_entries+31ULL)/32ULL;
        for(unsigned long long w=0;w<words&&spill_entry==0ULL;++w){
          unsigned int seen=spill_map[w];
          for(unsigned int b=0;b<32u&&spill_entry==0ULL;++b){
            if(w*32ULL+b>=spill_entries)break;
            if(seen&(1u<<b))continue;
            seen=atomicOr(spill_map+w,1u<<b);
            if((seen&(1u<<b))==0u)spill_entry=w*32ULL+b+1ULL;
          }
        }
        if(spill_entry==0ULL){gf_trap();return nullptr;}  // more columns spilling at once than entries
      }
      const unsigned long long entry=spill_entry-1ULL;
      char* block=spill_pool+(entry/GF_LANES)*spill_bytes*GF_LANES+4ULL*(entry%GF_LANES);
      void* result=block+(position-capacity)*GF_LANES;position+=n;highwater=gf_max(highwater,position);return result;
#endif
    }
    void* result=data+position*GF_LANES;position+=n;highwater=gf_max(highwater,position);return result;
  }
  __device__ void* allocate_bytes(unsigned long long n){
    n=(n+15)&~15ULL;
    if(bposition+n>bcapacity){gf_trap();return nullptr;}
    void* result=bytes+bposition;bposition+=n;return result;
  }
};
struct GFFrame {
 GFWorkspace &ws;unsigned long long saved,bsaved;
 __device__ GFFrame(GFWorkspace &w):ws(w),saved(w.position),bsaved(w.bposition){}
 __device__ ~GFFrame(){ws.position=saved;ws.bposition=bsaved;}
};'''

#: The column entry's call of the WRF driver, with its pointer arguments.
_CALL = re.compile(r'^ cu_grell_freitas_li\(workspace, (.*)\);$', re.M)
_ARGUMENT = re.compile(r'\((GFReal|int)\*\)\(ptrs\[(\d+)\]\+column\*strides\[\d+\]\)')


def _pointer_plus(scheme):
    """``((GFReal *) ((char *) X + D))`` -> ``gf_lane_plus<GFReal>(X, D)`` (balanced D)."""
    head = re.compile(r'\(\(GFReal \*\) \(\(char \*\) ([A-Za-z_][\w.>-]*) \+ ')
    out, pos, count = [], 0, 0
    while (m := head.search(scheme, pos)):
        depth, j = 2, m.end()
        while depth:
            if scheme[j] == '(':
                depth += 1
            elif scheme[j] == ')':
                depth -= 1
            j += 1
        # scheme[m.end():j-2] is D; the two closers end the (char *) sum and the cast.
        displacement = scheme[m.end():j - 2]
        assert scheme[j - 2:j] == '))', scheme[m.start():j]
        out.append(scheme[pos:m.start()] + 'gf_lane_plus<GFReal>(' + m[1] + ', ' + displacement + ')')
        pos, count = j, count + 1
    out.append(scheme[pos:])
    return ''.join(out), count


def lower(support, scheme, entry, metal=False):
    """Return the lane-interleaved CUDA source (support, scheme, lane entry) from the pinned pieces.

    ``metal``: the Metal lowering (:func:`metal_locals`), the same operations in
    the same order with every local whose address is passed on held in device
    memory; the CUDA text is unchanged by it.
    """
    assert support.count(_WORKSPACE_OLD) == 1, 'GFL workspace support changed'
    support = support.replace(_WORKSPACE_OLD, _WORKSPACE_NEW)
    assert support.count('typedef char GFText[50];') == 1
    support = support.replace('typedef char GFText[50];', 'typedef char GFText[50];\n' + _LANE_POINTER, 1)
    # The lane entry: the same call, each pointer argument the column's word in its field.
    call = _CALL.search(entry)
    assert call, 'GFL column entry changed'
    arguments, count = _ARGUMENT.subn(lambda m: f'({m[1]}*)(gfl_field(ptrs,strides,{m[2]},col))', call[1])
    assert count == 46, count
    defines, lane_entry = _DEFINES, _ENTRY.replace('@ARGUMENTS@', arguments)
    lowered = lane_scheme(scheme)
    if metal:
        lowered, need = metal_locals(lowered)
        lane_entry, bounds = _metal_bounds(lane_entry)
        # ierrc/ierrcm/ierrcs (192), fscav (16), the promoted locals (nested calls at most add up) and the bounds
        assert 192 + 16 + need + bounds <= METAL_CHARACTER_BYTES, need
        defines = defines.replace('#define GF_CHARACTER_BYTES %d\n' % CHARACTER_BYTES,
                                  '#define GF_CHARACTER_BYTES %d\n' % METAL_CHARACTER_BYTES)
        # Every lane pointer now points into device memory (scratch, fields, the byte region).
        assert support.count(_LANE_FIELD) == 1
        support = support.replace(_LANE_FIELD, _LANE_FIELD_METAL)
    return defines + support + '\n' + lowered + '\n' + lane_entry


#: Scalar and array types of the scheme's locals.
_LOCAL_TYPES = {'GFReal': 4, 'int': 4, 'bool': 1, 'long long': 8, 'unsigned int': 4}
#: A scheme function: every one takes the column workspace first.
_SCHEME_FUNCTION = re.compile(r'^__device__ [^\n;]*\(GFWorkspace &workspace[^;{]*\)\s*\n\{\n', re.M)
_FUNCTION_END = re.compile(r'^\}\n', re.M)
_LOCAL_TYPE = '|'.join(re.escape(t) for t in _LOCAL_TYPES)


def _address_taken(body):
    """Names after an address-of ``&`` (as written in the lowered scheme), not the right side of ``&&``."""
    names = set()
    for m in re.finditer(r'&\s*(\w+)', body):
        at = m.start()
        if body[at + 1:at + 2] == '&' or body[at - 1:at] == '&':
            continue
        if body[at - 1:at].isalnum() or body[at - 1:at] in ('_', '.', '>'):
            continue
        names.add(m[1])
    return names


def metal_locals(scheme):
    """Hold every local whose address a scheme function passes on in the thread's contiguous region.

    On CUDA a lane pointer or descriptor may point at a local (``&itf``, a
    Fortran constant ``&C_7360``, ``zuh2``) or at the scratch: generic
    addressing covers both. A Metal pointer has one address space, so here
    each such local of a function becomes a reference to a 16-byte slot of
    one block the function takes from the workspace's byte region on entry
    (released on return by its own GFFrame). An initialised local is
    assigned its initialiser where it was declared. Only placement moves.
    Returns the lowered scheme and the bytes all functions take together
    (an upper bound on what nested calls hold at once).
    """
    out, pos, total = [], 0, 0
    for head in list(_SCHEME_FUNCTION.finditer(scheme)):
        end = _FUNCTION_END.search(scheme, head.end()).end()
        body = scheme[head.end():end]
        names = _address_taken(body)
        arrays = set(re.findall(r'^\s+(?:%s) (\w+)\[\d+\]' % _LOCAL_TYPE, body, re.M))
        names |= {a for a in arrays if re.search(r'[(,]\s*%s\s*[,)]' % re.escape(a), body)}
        offset = 0

        def slot(m):
            nonlocal offset
            indent, kind, name, count, value = m[1], m[2], m[3], m[4], m[5]
            at = offset
            offset += (_LOCAL_TYPES[kind] * (int(count) if count else 1) + 15) & ~15
            if count:
                assert value is None, name
                return f'{indent}{kind} (&{name})[{count}] = *({kind} (*)[{count}]) (gf_locals + {at});'
            line = f'{indent}{kind} &{name} = *({kind} *) (gf_locals + {at});'
            return line if value is None else f'{line} {name} = {value};'

        if names:
            alternatives = '|'.join(sorted(map(re.escape, names), key=len, reverse=True))
            declaration = re.compile(r'^(\s+)(%s) (%s)(?:\[(\d+)\])?(?: = ([^;]*))?;$' % (_LOCAL_TYPE, alternatives), re.M)
            body = declaration.sub(slot, body)
        if offset:
            body = ('  GFFrame gf_locals_frame (workspace);\n'
                    f'  char *gf_locals = (char *) workspace.allocate_bytes ({offset});\n' + body)
            total += offset
        out.append(scheme[pos:head.end()] + body)
        pos = end
    out.append(scheme[pos:])
    return ''.join(out), total


_ENTRY_BOUNDS = re.compile(r'^ int (ims=1,[^;]*);$', re.M)


def _metal_bounds(entry):
    """The entry's WRF bounds (passed to the driver by address) in the thread's contiguous region."""
    m = _ENTRY_BOUNDS.search(entry)
    assert m, 'GFL entry bounds changed'
    pairs = [d.split('=', 1) for d in m[1].split(',')]
    lines = [f' int* gf_bounds=(int*)workspace.allocate_bytes({4 * len(pairs)});']
    lines += [f' int &{name}=gf_bounds[{i}];{name}={value};' for i, (name, value) in enumerate(pairs)]
    return entry[:m.start()] + '\n'.join(lines) + entry[m.end():], (4 * len(pairs) + 15) & ~15


_FSCAV_LOCAL = '  GFReal fscav[1];\n'
_FSCAV_SCRATCH = '  GFReal (&fscav)[1] = *(GFReal (*)[1]) workspace.allocate_bytes (16);\n'


_LANE_FIELD = '  T* p;\n'
_LANE_FIELD_METAL = '  T* p __attribute__((annotate("woof_space:device")));\n'


def lane_scheme(scheme):
    """The pinned transcription with lane pointers: the same text but for pointer spellings.

    tests/test_gfl_873b414.py spells the result back and requires the pinned
    _SCHEME_SOURCE byte for byte.
    """
    assert not re.search(r'\bG[FIB]P\b|gf_lane_plus|allocate_bytes', scheme), 'lane spellings already present'
    # Character arrays: their storage is the contiguous byte region.
    text = re.findall(r'\b\w+ = \(GFText \*\) (D_\d+);', scheme)
    assert len(text) == 3, text
    for name in text:
        pattern = name + ' = (char *) workspace.allocate ('
        assert scheme.count(pattern) == 1, name
        scheme = scheme.replace(pattern, name + ' = (char *) workspace.allocate_bytes (')
    # fscav, the driver's one-word local array, is passed by descriptor (data = &fscav[0]) to
    # cu_gfl_deep_run, whose other descriptors hold scratch words; a Metal struct pointer
    # field has one address space, so its word lives in the per-thread byte region as well.
    # A reference to the array keeps every use (sizeof, &fscav[0]) spelled and sized as before.
    assert scheme.count(_FSCAV_LOCAL) == 1
    scheme = scheme.replace(_FSCAV_LOCAL, _FSCAV_SCRATCH)
    scheme, plus = _pointer_plus(scheme)
    assert plus == 36, plus
    # Every float/integer/logical pointer in the scheme is a lane pointer. Fixed-size
    # local arrays stay C arrays: alpha/g_alpha/xk/xff_shal are read locally, zuh2 is
    # passed to get_zu_zd_pdf_fim, which never reads it, and fscav (one contiguous word, above)
    # is read at element 0 only (descriptor lbound = ubound = 1).
    before = len(re.findall(r'\bGFReal \*|\bint \*|\bbool \*', scheme))
    scheme = re.sub(r'\bGFReal \*', 'GFP ', scheme)
    scheme = re.sub(r'\bint \*', 'GIP ', scheme)
    scheme = re.sub(r'\bbool \*', 'GBP ', scheme)
    assert before > 1200 and not re.search(r'\bGFReal \*|\bint \*|\bbool \*', scheme), before
    return scheme


_ENTRY = r'''
// WRF v4.8.0 06d4240 phys/GFL 873b414/WRF/module_cu_gfl_wrf.F90:32-763 (one column per thread, WRF_CHEM=0),
// lane-interleaved (gpuwm/core/gfl_lanes.py). Field f of column c is at ptrs[f] + (c/GF_LANES)*strides[f]
// + (c%GF_LANES)*4 for a level field ([slab][level][lane]) or a surface field ([column], strides[f] =
// GF_LANES*4), and at ptrs[f] for a scalar (strides[f] = 0).
__device__ __forceinline__ char* gfl_field(const unsigned long long* ptrs,const unsigned long long* strides,int f,long long c){
 const unsigned long long s=strides[f];
 return (char*)ptrs[f]+(s==0?0ULL:(unsigned long long)(c/GF_LANES)*s+(unsigned long long)(c%GF_LANES)*4ULL);
}
__device__ void gfl_column(long long col,const unsigned long long* ptrs,const unsigned long long* strides,GFWorkspace &workspace,
 unsigned int* diagnostics,unsigned long long* highwater,const int* vertical,unsigned int* saved_fct_error){
#ifdef GF_HOST_WITNESS
 workspace.fct1d3_error=saved_fct_error[0]!=0;
#else
 workspace.fct1d3_error=atomicAdd(saved_fct_error,0u)!=0;
#endif
 int ims=1,ime=1,jms=1,jme=1,kms=vertical[0],kme=vertical[1],ids=1,ide=2,jds=1,jde=2,kds=vertical[2],kde=vertical[3],its=1,ite=1,jts=1,jte=1,kts=vertical[4],kte=vertical[5];
 cu_grell_freitas_li(workspace, @ARGUMENTS@);
 diagnostics[col]=workspace.diagnostics;highwater[col]=workspace.highwater;
 // WRF v4.8.0 06d4240 phys/GFL 873b414/module_cu_gfl_deep.F90:2641-2705 (fct1d3's SAVE error flag)
 if(workspace.fct1d3_error){
#ifdef GF_HOST_WITNESS
 saved_fct_error[0]=1;
#else
 atomicOr(saved_fct_error,1u);
#endif
 }
}
#ifndef GF_HOST_WITNESS
// A block is one slab of the scratch (blockDim = GF_LANES; thread t is lane t of the slab and keeps it; a
// column that spills holds a spill entry until it ends). next[0] is the column counter (gf_counter: 32-bit
// on Metal, where the launcher bounds it, 64-bit on CUDA); spill_map is the pool's entry bitmap. Each warp takes the next 32 consecutive columns from
// the launch's counter until none are left, so a warp held up by convecting columns does not leave the
// card idle behind it (a fixed column-per-slot round left the last round a few blocks wide). Columns are
// independent; each executes the complete source order from a fresh workspace, so which warp runs a
// column changes no word.
extern "C" __global__ void __launch_bounds__(GF_LANES, GFL_MIN_BLOCKS) gfl_lane_columns(long long ncol,
 const unsigned long long* ptrs,const unsigned long long* strides,char* scratch,unsigned long long capacity,
 unsigned long long spill_bytes,unsigned int* diagnostics,unsigned long long* highwater,const int* vertical,
 unsigned int* saved_fct_error,gf_counter* next,char* spill_pool,unsigned long long spill_entries,
 unsigned int* spill_map){
 const unsigned int lane=threadIdx.x, warp_lane=threadIdx.x&31u;
 char* slab=scratch+(unsigned long long)blockIdx.x*(capacity+GF_CHARACTER_BYTES)*GF_LANES;
 for(;;){
   gf_counter first=0;
   if(warp_lane==0)first=atomicAdd(next,(gf_counter)32);
   first=__shfl_sync(0xffffffffu,first,0);
   if(first>=(gf_counter)ncol)break;
   const long long col=(long long)first+warp_lane;
   if(col<ncol){
     GFWorkspace workspace{slab+4ULL*lane,capacity,0,0,0,GFReal(0.0f),false,spill_pool,spill_bytes,spill_entries,
       spill_map,0ULL,slab+capacity*GF_LANES+(unsigned long long)lane*GF_CHARACTER_BYTES,GF_CHARACTER_BYTES,0,lane};
     gfl_column(col,ptrs,strides,workspace,diagnostics,highwater,vertical,saved_fct_error);
     if(workspace.spill_entry!=0ULL){  // the column is done: its spill entry is free for the next one
       const unsigned long long e=workspace.spill_entry-1ULL;
       atomicAnd(spill_map+e/32ULL,~(1u<<(unsigned int)(e%32ULL)));
     }
   }
   __syncwarp();
 }
}
#else
extern "C" void gfl_lane_columns_host(long long ncol,const unsigned long long* ptrs,const unsigned long long* strides,
 char* scratch,unsigned long long capacity,unsigned long long spill_bytes,unsigned int* diagnostics,
 unsigned long long* highwater,const int* vertical,unsigned int* saved_fct_error,unsigned long long*){
 char characters[GF_CHARACTER_BYTES];
 for(long long c=0;c<ncol;c++){
   GFWorkspace workspace{scratch,capacity,0,0,0,GFReal(0.0f),false,nullptr,spill_bytes,0,nullptr,0,characters,
     GF_CHARACTER_BYTES,0,0};
   gfl_column(c,ptrs,strides,workspace,diagnostics,highwater,vertical,saved_fct_error);
 }
}
#endif
'''
