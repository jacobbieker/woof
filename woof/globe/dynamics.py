"""Moist hydrostatic spherical-harmonic dynamics on a hybrid pressure grid."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from woof.globe.spectral.diffusion import ExponentialHyperdiffusion
from woof.globe.spectral.eddy_viscosity import SpectralEddyViscosity
from woof.globe.spectral.fused import (
    momentum_bernoulli_kernel,
    scalar_tendency_kernel,
    tracer_flux_kernel,
    vertical_flux_divergence_kernel,
)
from woof.globe.spectral.timestep import step_with_scheme
from woof.globe.spectral.vector import VorticityDivergenceOperator

from .constants import (
    DRY_AIR_CP,
    DRY_AIR_GAS_CONSTANT,
    EARTH_ROTATION_RATE_S,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
    ADVECTED_TRACERS,
    CONDENSATE_SPECIES,
    GRID_TRACERS,
    NUMBER_MOMENTS,
    SPECTRAL_FIELDS,
    WATER_SPECIES,
    DEFAULT_EULERIAN_STEP_RULE_FRACTION,
)
from .bands import (
    BandPipeline,
    AssociativeAccumulator,
    LatitudeAccumulator,
    PlaneAccumulator,
    associative_over as _associative_over,
)
from .imex import DEFAULT_INTEGRATOR, IMEX_TABLEAUX, IMPLICIT_NEUTRAL_LIMIT, imex_step
from .physics.exchange import PhysicsExchange
from .profile import profiler_of
from .semi_implicit import BarotropicSemiImplicit, VerticalModeSemiImplicit
from .semilag.options import SemiLagrangianOptions
from .semilag.step import SEMILAG_INTEGRATORS, semilag_step
from .spill import SpilledArray, resident, spilled
from .state import ArwenGlobalState, MoistHybridState
from .transport import GridTracerTransport
from .vertical import HybridCoordinate
from .water import atmospheric_water_column, total_water_column


class SynthesisMemo:
    """Grid syntheses of a spectral state, keyed on the identity of the
    arrays that produced them.

    One model step synthesizes the same fields of the same state several
    times: the wind of the state entering the dynamics for the CFL gate,
    for the first stage's right-hand side and for the tracer transport's
    start fluxes; the wind, pressure and mass-flux divergence of the
    advanced state for the transport's end fluxes and again for the
    second physics exchange; the finished state's temperature, vapor and
    pressure for the in-situ ledger, for enforce() and for the next
    step's first physics exchange (profile 2026-09-04: 1,000 of the
    step's 5,600 field transforms at T255 were repeats).  The memo hands
    back the array a previous call computed when the spectral (or grid
    tracer) arrays it was computed from are the very same objects, so
    the value is the same bits by construction; a state whose field was
    replaced (a new array) misses on that field and recomputes it.

    Entries hold a reference to their source arrays so an object id can
    never be recycled under a live entry.  Memory: at most one entry per
    name, so the memo holds one grid state's worth of arrays at most,
    and the model clears it before every physics suite call (the step's
    device peak) so the peak is untouched.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[tuple, object, tuple]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(*arrays) -> tuple:
        return tuple(id(array) for array in arrays)

    def get(self, name: str, key: tuple):
        entry = self._entries.get(name)
        if entry is not None and entry[0] == key:
            self.hits += 1
            return entry[1]
        self.misses += 1
        return None

    def put(self, name: str, key: tuple, value, sources: tuple) -> None:
        self._entries[name] = (key, value, tuple(sources))

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def band_view(xp, value, rows: slice):
    """``value``'s latitude band as a CONTIGUOUS SLAB -- or ``value``
    itself when the band is the globe.

    A BAND IS A SLAB, NOT A STRIDE, and that is arithmetic rather than
    layout.  MEASURED 2026-09-06 on an RTX 5070 Ti, float32 and float64 at
    T255 and T533 shapes: a longitude reduction over a strided latitude
    slice of a levelled array returns different bits from the same rows in
    a contiguous array -- ``cupy.sum(v[:, r0:r1, :], axis=-1)`` differs
    from ``cupy.sum(v, axis=-1)[:, r0:r1]`` at every band count, while
    ``cupy.sum(ascontiguousarray(v[:, r0:r1, :]), axis=-1)`` is equal to
    it at every band count.  The reduction's kernel reads a contiguous
    operand one way and a strided one another; the band count never
    entered it.

    That defect reached the ten-step T255 checkpoint gate before this line
    existed: the lid absorber's four zonal means and the guards' vapor
    readings were handed strides, and steps 5 and 10 differed from the
    resident run in the wind, the vapor, the surface and the whole Noah
    namespace while every banded count agreed with every other.  So the
    band is materialised here, once, where it is cut -- which is also the
    slab the design's strided-copy measurement prices (full contiguous
    rate from four latitude rows upward).

    At one band the band IS the whole array, and handing back the array
    rather than a fresh view keeps object identity where the resident step
    relied on it (the synthesis memo serves one array, not two views of
    it), keeps its memory profile, and copies nothing.
    """
    if isinstance(value, SpilledArray):
        # The tier holds this one.  A staged band arrives on the card as
        # its own contiguous allocation, so it is already the slab the
        # paragraphs above require; nothing else about the band changes.
        return value.stage(rows)
    if rows.start == 0 and rows.stop == value.shape[-2]:
        return value
    return xp.ascontiguousarray(value[..., rows, :])


class BandedField:
    """A grid-space field the band loop reads one latitude band at a time.

    At ONE band it is the whole array the resident step already built, and
    a band is a view of it: the shipped default keeps the memory route it
    has today, and the band loop is a loop of one.  Above one band it is
    the Fourier waist the Legendre contraction produced, and a band is the
    inverse FFT of its rows -- the contraction ran whole either way, at
    ``N = nlat``, which is why the two return the same bits.
    """

    __slots__ = ("transform", "whole", "waist")

    def __init__(self, transform, *, whole=None, waist=None):
        if (whole is None) == (waist is None):
            raise ValueError("a banded field is either a whole array or a waist")
        self.transform = transform
        self.whole = whole
        self.waist = waist

    def band(self, rows: slice):
        if self.whole is not None:
            return band_view(self.transform.backend.xp, self.whole, rows)
        return self.transform.waist_band_to_grid(
            self.waist, rows.start, rows.stop
        )


class BandedStack(BandedField):
    """:class:`BandedField` for the stacked synthesis of several spectral
    fields, which is chunked at ``spectral_chunk`` and therefore carries
    one waist per chunk (the chunk is the Legendre GEMM's M dimension and
    MEASURED arithmetic, so the chunk boundaries are the resident call's).
    """

    __slots__ = ("names", "chunks", "nlev", "dtype")

    def __init__(self, transform, names, *, whole=None, chunks=(), nlev=0, dtype=None):
        self.transform = transform
        self.whole = whole
        self.waist = None
        self.names = tuple(names)
        self.chunks = tuple(chunks)
        self.nlev = int(nlev)
        self.dtype = dtype

    def band(self, rows: slice):
        if self.whole is not None:
            return band_view(self.transform.backend.xp, self.whole, rows)
        if len(self.chunks) == 1:
            return self.transform.waist_band_to_grid(
                self.chunks[0][1], rows.start, rows.stop
            )
        xp = self.transform.backend.xp
        out = xp.empty(
            (len(self.names), self.nlev, rows.stop - rows.start,
             self.transform.grid.nlon),
            dtype=self.dtype,
        )
        for start, waist in self.chunks:
            block = self.transform.waist_band_to_grid(
                waist, rows.start, rows.stop
            )
            out[start : start + block.shape[0]] = block
            del block
        return out


class GridSources:
    """Everything one grid-space pass over a spectral state needs, in the
    form the band loop reads it: the stacked synthesis, the two surface
    planes the hybrid pressures are built from, and the wind.

    The whole streaming decomposition lives in the distinction this type
    draws.  What is SMALL and shared by every band -- the ln ps plane, its
    exponential, the gradient of ln ps -- is built once and whole (5.13 MB
    a plane at T533).  What is a LEVELLED VOLUME is never built whole at
    all above one band: it is a waist two thirds its size, and the band
    that reads it drains its own rows.
    """

    __slots__ = ("state", "need", "stack", "pressure", "wind")

    def __init__(self, state, need, stack=None, pressure=None, wind=None):
        self.state = state
        self.need = need
        self.stack = stack
        self.pressure = pressure
        self.wind = wind


class _PhysicsSources:
    """What one physics call reads that is built ONCE per call and drained
    a band at a time: the syntheses, the continuity closure's mass-flux
    sources, the geometry planes, the model-top pressure read over the
    whole grid, and the accumulators the exchange filler and the water
    reading reduce through (dynamics.MoistHybridModel._physics_sources)."""

    __slots__ = ("sources", "div_mass", "lat", "lon", "holes", "water_rows",
                 "model_top_pa", "dt_s", "time_s")

    def __init__(self, *, sources, div_mass, lat, lon, holes, model_top_pa,
                 dt_s, time_s):
        self.sources = sources
        self.div_mass = div_mass
        self.lat = lat
        self.lon = lon
        self.holes = holes
        self.water_rows = None
        self.model_top_pa = model_top_pa
        self.dt_s = dt_s
        self.time_s = time_s

    def release(self) -> None:
        self.sources = None
        self.div_mass = None


class _GridAssembler:
    """The whole-grid destination of a state the physics returns a band at
    a time: the grid tracers, the surface reservoirs, the native physics
    namespace.

    Three destinations, and which one is chosen is the run's memory
    plan, not the arithmetic:

    - at ONE band the band is the globe, so the array is taken as it is
      and nothing is copied: the resident run's memory profile is the
      profile it had before the suite was banded;
    - a slice the pinned host tier holds is written into its slot band by
      band (:meth:`spill.SpilledArray.store` with rows), so the card never
      holds the whole array at all -- which is the relief the design's
      T533 and T799 rows are priced on;
    - otherwise a whole device buffer is allocated at the first band and
      filled in place.

    A slot the tier does not yet have (a namespace array the suite seeds
    on its first call) is opened at its whole shape and filled the same
    way.  The band's rows are the band's rows: nothing here reads a row
    it was not handed, so the assembled array is the array the resident
    call returns, byte for byte.
    """

    def __init__(self, model, prefix: str, targets, *, parked: bool):
        self.xp = model.transform.backend.xp
        self.nlat = int(model.transform.grid.nlat)
        self.resident = bool(model.pipeline.resident)
        self.tier = model.host_tier if parked else None
        self.prefix = str(prefix)
        self.targets = dict(targets or {})
        self.out: dict[str, object] = {}

    def put(self, name: str, rows: slice, value) -> None:
        if self.resident:
            self.out[name] = value
            return
        shape = (*value.shape[:-2], self.nlat, int(value.shape[-1]))
        if self.tier is not None:
            handle = self.out.get(name)
            if handle is None:
                target = self.targets.get(name)
                if (
                    isinstance(target, SpilledArray)
                    and target.tier is self.tier
                    and target.shape == shape
                    and target.dtype == np.dtype(value.dtype)
                ):
                    handle = target
                else:
                    handle = self.tier.open(
                        f"{self.prefix}__{name}", shape, value.dtype)
                self.out[name] = handle
            handle.store(value, rows)
            return
        buffer = self.out.get(name)
        if buffer is None:
            buffer = self.out[name] = self.xp.empty(shape, dtype=value.dtype)
        buffer[..., rows, :] = value

    def finish(self) -> dict[str, object]:
        if self.resident and self.tier is not None:
            return self.tier.hold(self.prefix, self.out)
        return self.out


def spectral_cfl_refusal(cfl: float, dt_s: float, maximum_cfl: float,
                         truncation: int, radius_m: float) -> str:
    """The Eulerian CFL refusal, worded so a day stronger than any the
    shipped step was set against is explained and not just stopped: the
    step this flow admits at the gate, the step it admits at the shipped
    rule (constants.DEFAULT_EULERIAN_STEP_RULE_FRACTION of the gate), and
    the maximum wind the reading implies through the gate's own formula
    dt |V|max sqrt(N(N+1)) / a.  The receipt's ``cfl`` block carries the
    same numbers (runner.cfl_headroom_receipt)."""
    rate = cfl / dt_s
    rule = DEFAULT_EULERIAN_STEP_RULE_FRACTION
    wind = rate * radius_m / math.sqrt(truncation * (truncation + 1.0))
    return (
        f"WOOF global spectral CFL {cfl:.3f} exceeds {maximum_cfl:.3f} "
        f"at dt_s = {dt_s:g} s: this flow admits dt_s <= {maximum_cfl / rate:.1f} s "
        f"at the gate and <= {rule * maximum_cfl / rate:.1f} s at the shipped "
        f"{rule:g}-of-gate rule (implied maximum wind {wind:.1f} m/s); "
        "reduce dt_s or truncation"
    )


@dataclass
class MoistHybridModel:
    transform: object
    vertical: HybridCoordinate
    surface_geopotential: object
    physics: object | None = None
    rotation_rate_s: float = EARTH_ROTATION_RATE_S
    gas_constant: float = DRY_AIR_GAS_CONSTANT
    cp: float = DRY_AIR_CP
    integrator: str = DEFAULT_INTEGRATOR
    # How the physics suite composes with the dynamics over one step.
    # "strang": the suite over dt/2 before the dynamics and over dt/2 after
    # it, the symmetric split every archive so far was written under.
    # "merged": the suite ONCE per step, over the full dt, after the
    # dynamics.  Strang applies the trailing half of one step and the
    # leading half of the next back to back, with only the positivity
    # repair, the water fixer and the guards between them; merged runs
    # that pair as one call.  What differs from Strang: the cold start's
    # leading half step (dt/2 of physics at t = 0) is never applied, the
    # run's final state carries a full dt of physics where Strang's
    # carries dt/2, and each scheme integrates its rates over dt instead
    # of dt/2 (one call of dt against two of dt/2 differs by the schemes'
    # own nonlinearity over dt/2, the order of the split error itself);
    # every scheme then runs at the model's step, the step WRF runs the
    # same schemes at (about 300 s at 52 km).  The physics costs half:
    # one suite call, one exchange, one return and one positivity pass
    # per step instead of two of each (the two suite calls were 53
    # percent of the step at T255, profile 2026-09-04).  The energy
    # ledger's physics_first and positivity_first marks read zero under
    # merged; the receipt names the split.
    physics_split: str = "strang"
    diffusion: ExponentialHyperdiffusion | None = None
    divergence_diffusion_strength: float = 1.5
    pressure_diffusion_strength: float = 0.25
    # Applies to water VAPOR, the one water field in the spectral basis.
    # The condensate species and the number moments are grid tracers
    # (constants.GRID_TRACERS) carried by the positive-definite flux-form
    # transport, whose limiter is their only diffusion: an explicit
    # spectral hyperdiffusion of a rain shaft was part of the ringing the
    # 2026-09-02 finding traced, and a grid-point tracer has no spectral
    # coefficients to diffuse.
    water_diffusion_strength: float = 0.5
    # Every vertical gravity-wave mode implicit by default (fixed means
    # default; audit 2026-09-01 DN-1); BarotropicSemiImplicit stays
    # selectable for identity-locked archives.
    semi_implicit: VerticalModeSemiImplicit | BarotropicSemiImplicit = (
        VerticalModeSemiImplicit()
    )
    mass_fixer: bool = True
    water_fixer: bool = True
    positivity_repair: bool = True
    maximum_cfl: float = 0.75
    # [sponge] Top-of-model graded wave absorber.  This is a DYCORE
    # remedy: the rigid p_top lid reflects terrain-locked near-truncation
    # wave trains (T533 diagnosis 2026-08-31) and the absorber stands in
    # for the absorption the lid cannot provide, so it lives with the lid
    # and acts in EVERY physics mode.  Its previous home inside the
    # reference suite (v7) left every other mode unprotected: the full
    # native five-scheme T255 run died at hour 6.55 on the 140 K research
    # bound (polar-top collapse, 242 m/s top winds) with the absorber
    # configured and never acting.  Semantics unchanged from suite v7:
    # rings whose ring-mean p_full sits below sponge_base_pa Rayleigh-damp
    # their u and v anomalies toward the instantaneous zonal mean, at a
    # rate that ramps as cos^2 from zero at the base to
    # 1/sponge_lid_relaxation_time_s at the model lid.  Sizing arithmetic
    # (T533 hour-3 crash states, 2026-08-31): the pocket amplitude
    # e-folds in 4230 s (2.36e-4 1/s); the lid rate 1/900 s = 1.11e-3
    # 1/s beats it 4.7x (~2.4x after the ramp halves the depth average).
    # sponge_base_pa=0 disables.
    sponge_base_pa: float = 5000.0
    sponge_lid_relaxation_time_s: float = 900.0
    initial_provenance: dict | None = None
    # Widest stack a single transform call may carry.  Full-width stacks at
    # T533 fp32 materialize ~2.5 GB complex Fourier temporaries per operand
    # and ran a 32 GB RTX 5090 out of memory; chunks of six keep the batching
    # win (the per-order loop still runs once per chunk, not once per field)
    # while bounding peak memory.
    #
    # THE GUARD'S STANDING, 2026-09-06, after the transform split at the
    # Fourier waist.  The temporaries this ceiling bounds are the waist
    # itself and the live rfft/irfft buffers either side of it.  The
    # default stays six, and the ceiling is now CONSERVATIVE rather than
    # exact: even at the shipped one band the operand that survives the
    # fill and lives through both GEMMs is the waist, (*lead, nlat,
    # T+1), and the (*lead, nlat, nlon//2+1) scaled spectrum the
    # paragraph above prices does not survive it at all.  That is 0.666
    # of the width it was -- 0.765 against 1.149 GiB for a six-field
    # forty-level stack at T533 in single precision -- and the fill
    # table in SphericalHarmonicTransform.fourier_waist carries the
    # measured peak that fell with it.  Above one band the widest of
    # those buffers divides with the band count as well, so a banded run
    # RE-MEASURES this ceiling rather than inheriting it.  It cannot
    # simply be widened on that margin: the chunk is the Legendre GEMM's
    # M dimension and MEASURED 2026-09-06 to move one ulp of a synthesis
    # at two-, five- and ten-level ladders, so it carries its own config
    # identity and moving the default would move the arithmetic of every
    # shipped config.  Lane 5's guard sweep owns the widening, with the
    # margin above for it and that identity cost against it.
    spectral_chunk: int = 6
    # In-situ observer slot (woof.globe.insitu): called once per
    # step with the finished state before enforce().  Read-only by contract.
    observer: object | None = None
    # Step profiler slot (woof.globe.profile): every operator of
    # the step is wrapped in a named section; None makes every hook a
    # no-op context manager and the step bit-identical to the unhooked one.
    profiler: object | None = None
    # The synthesis memo (SynthesisMemo): repeated syntheses of one state
    # within a step are served from it.  Same bits either way; False
    # recomputes every synthesis (the memory-tightest form).
    synthesis_memo: bool = True
    # The Lipschitz ceiling of the semi-Lagrangian core (semilag/), read
    # by no other integrator.  It REPLACES the advective CFL gate on that
    # path: a semi-Lagrangian advection has no advective CFL limit, and
    # what does bound it is the flow deformation (semilag.trajectory).
    maximum_lipschitz: float = 0.75
    # The [semilag] options of the same core; defaults everywhere else.
    semilag: SemiLagrangianOptions = SemiLagrangianOptions()
    # How many latitude bands grid space is streamed through (bands.py).
    # ONE is the resident run and the shipped default; above one, every
    # grid-space operator of the step runs a band at a time while spectral
    # space stays whole, so the Legendre contraction keeps K = N = nlat
    # and the answer does not move (gates BIT-1 to BIT-4).  It is a
    # streaming granularity, not a decomposition: it enters no identity
    # hash and no checkpoint field.
    latitude_bands: int = 1
    # The pinned host tier (spill.HostTier), and which slices of the
    # persistent grid state live in it.  None is the resident run and the
    # shipped default: the tracers, the surface reservoirs and the native
    # physics namespace stay on the card exactly as they do today.  With a
    # tier, those slices live in pinned host memory and reach the card
    # only as the copy their consumer was going to make anyway, so the
    # card stops holding the original beside it for the whole run.  Memory
    # only: a parked array holds the same bytes and every kernel reads the
    # same values (gate BIT-1 with spill on).
    host_tier: object | None = None
    spill_slices: tuple[str, ...] = ()
    #: The multi-card row exchange, or None for the shipped single-card
    #: run.  It changes WHICH bands this process executes and nothing
    #: about the schedule, so it appears in no identity and no pin (the
    #: ``gather`` mode's whole argument, cards.py).
    cards: object = None
    #: Latitude rows exchanged across a card boundary for the meridional
    #: tracer sweep's deep halo.  Unread on one card.
    card_halo_rows: int = 16

    def __post_init__(self) -> None:
        # Zero is the absorber's off switch, so it is not folded into the
        # nonnegativity check; a negative base selects no ring while
        # reading as a configured absorber - a run believed sponged then
        # dies exactly like the unsponged T533 runs (139.6-140.0 K at
        # hours 3.0-3.5, twice).
        if self.sponge_base_pa < 0.0:
            raise ValueError(
                "sponge base_pa must be nonnegative: a negative base "
                "selects no ring while reading as a configured absorber"
            )
        # A zero lid time divides the Rayleigh rate by zero; a negative
        # one turns the per-step factor exp(-dt/tau) into growth, so the
        # absorber would amplify the near-truncation wave train it exists
        # to remove.
        if self.sponge_lid_relaxation_time_s <= 0.0:
            raise ValueError(
                "sponge lid_relaxation_time_s must be positive: a "
                "nonpositive lid time makes exp(-dt/tau) amplify the wave "
                "anomalies instead of damping them"
            )
        if str(self.physics_split) not in ("strang", "merged"):
            raise ValueError(
                f"physics_split must be 'strang' or 'merged', got "
                f"{self.physics_split!r}"
            )
        self.vector = VorticityDivergenceOperator(self.transform)
        b = self.transform.backend
        self.surface_geopotential = b.asarray(
            self.surface_geopotential, dtype=b.float_dtype
        )
        self.transform._validate_grid(self.surface_geopotential)
        self._sinlat = b.asarray(
            self.transform.grid.sin_lat[:, None], dtype=b.float_dtype
        )
        self.coriolis = 2.0 * float(self.rotation_rate_s) * self._sinlat
        self._delta_b = b.asarray(self.vertical.delta_b, dtype=b.float_dtype)
        self._b_upper = b.asarray(self.vertical.b_half[:-1], dtype=b.float_dtype)
        self._b_lower = b.asarray(self.vertical.b_half[1:], dtype=b.float_dtype)
        self._target_mass_pa: float | None = None
        self._target_total_water_kg_m2: float | None = None
        # The grid tracers' transport (transport.GridTracerTransport):
        # directionally split, van Leer limited, sub-cycled to a 0.25
        # per-face Courant number, positive by construction.
        self.transport = GridTracerTransport(self.transform, self.vertical)
        # One schedule for the whole step: the transport reads the model's
        # rather than owning a second one, so a band edge is the same row
        # in every operator and a reduction buffer's layout is a pure
        # function of (nlat, bands).
        self.pipeline = BandPipeline(
            self.transform.grid.nlat, int(self.latitude_bands),
            exchange=self.cards,
        )
        self.transport.pipeline = self.pipeline
        self.transport.card_halo_rows = int(self.card_halo_rows)
        # The waist is the ONE place the analysis crosses a card boundary,
        # and the transform is the object that owns it.  The hook is
        # duck-typed rather than imported because woof.globe.spectral is
        # the layer below this one and must not learn that cards exist.
        if self.cards is not None:
            self.transform.row_exchange = self.cards
        self._memo = SynthesisMemo() if self.synthesis_memo else None
        # The band the derived-field cache holds, and what it holds for it
        # (_band_value).  At one band the tag never changes and the cache
        # is the whole-grid memo the resident step has always had.
        self._band_tag: tuple[int, int] | None = None
        self._band_values: dict[str, tuple] = {}

    def _chunked(self, fn, array):
        """Apply fn along leading-axis chunks of at most spectral_chunk.

        Each chunk result is copied into one preallocated stack as soon
        as it exists and then dropped, so at most one chunk result is
        live beside the stack.  Memory only, same bits: the former
        ``concatenate`` of a list of chunk results was the same copy,
        taken after every result had been held (2.29 + 2.29 GiB for a
        twelve-field T533 float32 synthesis).
        """
        limit = max(1, int(self.spectral_chunk))
        if array.shape[0] <= limit:
            return fn(array)
        xp = self.transform.backend.xp
        out = None
        for start in range(0, array.shape[0], limit):
            piece = fn(array[start : start + limit])
            if out is None:
                out = xp.empty(
                    (array.shape[0], *piece.shape[1:]), dtype=piece.dtype
                )
            out[start : start + piece.shape[0]] = piece
            del piece
        return out

    @property
    def nlev(self) -> int:
        return self.vertical.nlev

    def validate_state(self, state: MoistHybridState) -> None:
        volume = (self.nlev, *self.transform.spectral_shape)
        for name in SPECTRAL_FIELDS:
            value = getattr(state, name)
            expected = self.transform.spectral_shape if name == "log_surface_pressure" else volume
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} shape {value.shape} != {expected}")
        grid_volume = (self.nlev, *self.transform.grid.shape)
        for name in GRID_TRACERS:
            value = getattr(state, name)
            if value is None:
                raise ValueError(
                    f"grid tracer {name} is missing: every prognostic state "
                    "carries the ten condensate and number-moment fields as "
                    f"real {grid_volume} arrays on the Gaussian grid"
                )
            if tuple(value.shape) != grid_volume:
                raise ValueError(
                    f"grid tracer {name} shape {tuple(value.shape)} != {grid_volume}"
                )
            if value.dtype.kind == "c":
                raise TypeError(
                    f"grid tracer {name} is complex: the condensate species "
                    "and number moments are grid-point fields, not spectral "
                    "coefficients (checkpoint schema v3)"
                )

    def _hydrostatic_geopotential(
        self, virtual_temperature, p_half, p_full=None, ln_ratio=None,
        rows: slice | None = None,
    ):
        xp = self.transform.backend.xp
        if p_full is None:
            p_full = xp.sqrt(p_half[:-1] * p_half[1:])
        if ln_ratio is None:
            ln_ratio = xp.log(p_half[1:] / p_half[:-1])
        # phi at the lower interface of layer k is the surface geopotential
        # plus the hydrostatic thickness of every layer below; a reversed
        # cumulative sum computes all interfaces at once where the k-loop
        # was one kernel launch per level on a GPU backend.
        thickness = self.gas_constant * virtual_temperature * ln_ratio
        below = xp.flip(
            xp.cumsum(xp.flip(thickness[1:], axis=0), axis=0), axis=0
        )
        # The terrain is a plane, so a band adds its own rows of it; the
        # cumulative sum above it runs DOWN a column and is band-local.
        surface = (
            self.surface_geopotential if rows is None
            else self.surface_geopotential[rows]
        )
        lower_phi = xp.concatenate(
            [below, xp.zeros_like(thickness[:1])], axis=0
        ) + surface
        # The half layer above the interface integrates Tv linear in ln p
        # (HybridCoordinate.half_layer_thickness): the former level-only
        # form omitted R (dTv/dlnp) dlnp^2/8, whose terrain-following
        # variation was 1.06 m/s per hour of spurious top-level
        # acceleration at rest over a 2 km mountain (audit 2026-09-01,
        # DN-3).  It pairs with the midpoint pressure gradient in rhs().
        return lower_phi + self.vertical.half_layer_thickness(
            virtual_temperature, p_full, ln_ratio, xp,
            gas_constant=self.gas_constant,
        )

    _PRESSURE_KEYS = frozenset({
        "logps", "ps", "p_half", "p_full", "dp", "ln_ratio",
    })
    _GRID_KEYS = frozenset({
        "logps", "ps", "p_half", "p_full", "dp", "ln_ratio",
        "theta", "temperature", "virtual_temperature", "vorticity",
        "divergence", "u", "v", "geopotential", *ADVECTED_TRACERS,
    })

    def grid_state(
        self, state: MoistHybridState, only=None
    ) -> dict[str, object]:
        """Synthesize the grid-space view of a spectral state.

        ``only`` names the keys the caller actually reads; the dependency
        closure of just those keys is computed, so a caller that needs the
        water species and dp does not pay for winds, geopotential, or the
        vorticity/divergence syntheses.  Per-field values are identical to
        the full dictionary's - subsetting only avoids work.

        This is the WHOLE-GLOBE call: it asks the band loop for one band
        covering every row.  A caller that can work a band at a time asks
        :meth:`grid_sources` and :meth:`grid_band` instead and never
        materialises a levelled volume at full size.
        """
        g, _stack, _names = self._grid_state_stacked(state, only)
        return g

    def _grid_state_stacked(self, state: MoistHybridState, only=None):
        """:meth:`grid_state` plus the stacked synthesis it was built from.

        Returns ``(g, stack, names)``: ``stack`` is the one contiguous
        ``(len(names), nlev, nlat, nlon)`` array every 3-D spectral field
        in ``g`` is a row view of (``None`` when none was requested) and
        ``names`` its row order.  Callers that need those fields as a
        stack -- the positivity repair, the physics exchange, the scalar
        advection -- slice it instead of re-stacking the views, which at
        T533 float32 was a 2.1-2.3 GiB copy per call.  Same values by
        construction; the rows ARE the views.
        """
        sources = self.grid_sources(state, only)
        return self.grid_band(sources, self.whole_rows)

    @property
    def whole_rows(self) -> slice:
        """The band that is the globe: what a whole-grid caller asks for."""
        return slice(0, self.transform.grid.nlat)

    def _grid_closure(self, only):
        """The dependency closure of the requested keys, and the spectral
        fields one stacked synthesis has to carry to serve it."""
        if only is None:
            need = set(self._GRID_KEYS)
        else:
            need = set(only)
            unknown = need - self._GRID_KEYS
            if unknown:
                raise ValueError(
                    f"unknown grid_state keys {sorted(unknown)}; "
                    f"known keys are {sorted(self._GRID_KEYS)}"
                )
        if "geopotential" in need:
            need |= {"virtual_temperature", "p_half"}
        if "virtual_temperature" in need:
            need |= {"temperature", "qv", "qc", "qr", "qi", "qs", "qg"}
        if "temperature" in need:
            need |= {"theta", "p_full"}
        if need & self._PRESSURE_KEYS:
            need |= self._PRESSURE_KEYS
        spectral_names = tuple(
            name
            for name in ("theta", "qv", "vorticity", "divergence")
            if name in need
        )
        return need, spectral_names

    def grid_sources(self, state: MoistHybridState, only=None) -> GridSources:
        """The band loop's view of one spectral state's grid space.

        Every Legendre contraction the pass needs runs HERE, whole, at the
        operand shapes a resident step gives it; what comes back is either
        the resident grid arrays or the waists the bands drain.  Nothing
        below this line ever depends on the band count except how many
        rows it asks for at a time.
        """
        self.validate_state(state)
        need, spectral_names = self._grid_closure(only)
        return GridSources(
            state=state,
            need=need,
            stack=self._synthesis_stack(state, spectral_names)
            if spectral_names else None,
            pressure=self._pressure_sources(state)
            if need & self._PRESSURE_KEYS else None,
            wind=self._wind_sources(state) if need & {"u", "v"} else None,
        )

    def _synthesis_stack(self, state, spectral_names) -> BandedStack:
        """One stacked synthesis of the requested spectral fields.

        The batched contraction inside runs once per CALL, so separate
        calls would pay it once per field (and on a GPU backend, per-field
        rounds of kernel launches) for the same arithmetic.  The memo
        serves a stack a previous call computed from the same arrays when
        it holds every requested name; a partial hit recomputes the whole
        request so the stack handed back always carries the requested
        names in their canonical order (the advection slices consecutive
        rows of it).
        """
        xp = self.transform.backend.xp
        memo = self._memo
        sources = tuple(getattr(state, name) for name in spectral_names)
        key = SynthesisMemo.key(*sources)
        cached = memo.get("stack", key) if memo is not None else None
        if cached is not None and cached.names == tuple(spectral_names):
            return cached
        stacked = xp.stack(sources)
        if self.pipeline.resident:
            payload = BandedStack(
                self.transform, spectral_names,
                whole=self._chunked(self.transform.inverse, stacked),
                nlev=self.nlev, dtype=self.transform.backend.float_dtype,
            )
        else:
            limit = max(1, int(self.spectral_chunk))
            chunks = []
            for start in range(0, stacked.shape[0], limit):
                chunks.append((start, self.transform.contract_to_waist(
                    stacked[start : start + limit], bands=self.pipeline.bands
                )))
            payload = BandedStack(
                self.transform, spectral_names, chunks=chunks, nlev=self.nlev,
                dtype=self.transform.backend.float_dtype,
            )
        del stacked
        if memo is not None:
            memo.put("stack", key, payload, sources)
        return payload

    def _pressure_sources(self, state):
        """The synthesized ln ps and its exponential, and -- at one band --
        the hybrid pressure volumes they imply.

        The two PLANES are 5.13 MB each at T533 and shared by every band;
        the four VOLUMES they imply are 0.8 GiB and are built at the
        band's width above one band, because ``vertical.pressure`` is
        column-local and a band's rows are the rows the whole call would
        have written there.
        """
        memo = self._memo
        key = SynthesisMemo.key(state.log_surface_pressure)
        cached = memo.get("pressure", key) if memo is not None else None
        if cached is not None:
            return cached
        xp = self.transform.backend.xp
        logps = self.transform.inverse(state.log_surface_pressure)
        ps = xp.exp(logps)
        block = {"logps": logps, "ps": ps}
        if self.pipeline.resident:
            block.update(self.vertical.pressure(ps, self.transform.backend))
        if memo is not None:
            memo.put("pressure", key, block, (state.log_surface_pressure,))
        return block

    def _pressure_band(self, sources, rows: slice) -> dict[str, object]:
        block = {
            "logps": band_view(self.transform.backend.xp, sources["logps"], rows),
            "ps": band_view(self.transform.backend.xp, sources["ps"], rows),
        }
        if "p_half" in sources:
            for name in ("p_half", "p_full", "dp", "ln_ratio"):
                block[name] = band_view(self.transform.backend.xp, sources[name], rows)
            return block
        block.update(
            self.vertical.pressure(block["ps"], self.transform.backend)
        )
        return block

    def _pressure_block(self, state: MoistHybridState) -> dict[str, object]:
        """The whole-globe pressure block, for callers outside the band loop."""
        return self._pressure_band(self._pressure_sources(state), self.whole_rows)

    def _wind_sources(self, state: MoistHybridState):
        """The grid wind of a spectral state (vector.wind_from_vordiv), or
        the two gradient waists a band pipeline drains it from."""
        memo = self._memo
        sources = (state.vorticity, state.divergence)
        key = SynthesisMemo.key(*sources)
        cached = memo.get("wind", key) if memo is not None else None
        if cached is None:
            if self.pipeline.resident:
                cached = ("whole", self.vector.wind_from_vordiv(
                    state.vorticity, state.divergence
                ))
            else:
                xp = self.transform.backend.xp
                # The same two inverse Laplacians and the same stacked
                # gradient the resident call makes; only the drain waits.
                psi = self.transform.inverse_laplacian(state.vorticity)
                chi = self.transform.inverse_laplacian(state.divergence)
                cached = ("waists", self.transform.gradient_waists(
                    xp.stack([psi, chi]), bands=self.pipeline.bands
                ))
                del psi, chi
            if memo is not None:
                memo.put("wind", key, cached, sources)
        return cached

    def _wind_band(self, sources, rows: slice):
        kind, payload = sources
        if kind == "whole":
            u, v = payload
            xp = self.transform.backend.xp
            return band_view(xp, u, rows), band_view(xp, v, rows)
        d_east, d_north = self.transform.gradient_band(
            payload, rows.start, rows.stop
        )
        u = -d_north[0] + d_east[1]
        v = d_east[0] + d_north[1]
        return u, v

    def _wind(self, state: MoistHybridState):
        return self._wind_band(self._wind_sources(state), self.whole_rows)

    def _band_value(self, name: str, key, rows: slice, build):
        """Memoize one derived grid field for the band being processed.

        The step reads temperature, virtual temperature and the
        geopotential several times per band; holding one entry PER BAND
        would rebuild the whole-grid array the band count exists to
        divide, so the cache holds the band it is on and drops the band it
        left.  At one band the band never changes, so this is exactly the
        whole-grid memo the resident step has always had.
        """
        if self._memo is None:
            return build()
        tag = (rows.start, rows.stop)
        if tag != self._band_tag:
            self._band_values.clear()
            self._band_tag = tag
        entry = self._band_values.get(name)
        if entry is not None and entry[0] == key:
            return entry[1]
        value = build()
        self._band_values[name] = (key, value)
        return value

    def grid_band(self, sources: GridSources, rows: slice):
        """One latitude band of a state's grid space: ``(g, stack, names)``.

        Every expression below is the resident call's, evaluated on the
        band's rows.  Each of them reads only its own column or its own
        latitude row (the exner power, the virtual-temperature mixture,
        the hydrostatic cumsum over levels), so the band's values are the
        values the whole call would have written there.
        """
        state = sources.state
        need = sources.need
        g: dict[str, object] = {}
        stack = None
        names: tuple[str, ...] = ()
        if sources.stack is not None:
            names = sources.stack.names
            stack = sources.stack.band(rows)
            for index, name in enumerate(names):
                if name in need:
                    g[name] = stack[index]
        # The grid tracers ARE grid fields: handed over by reference (a
        # band by view), no synthesis, no ringing, and a consumer that
        # writes into one writes into the state (every consumer copies;
        # the bridge's fingerprint check proves the physics does).
        for name in GRID_TRACERS:
            if name in need:
                g[name] = band_view(self.transform.backend.xp, getattr(state, name), rows)
        if sources.pressure is not None:
            g.update(self._band_value(
                "pressure",
                SynthesisMemo.key(state.log_surface_pressure),
                rows,
                lambda: self._pressure_band(sources.pressure, rows),
            ))
        if "temperature" in need:
            key = SynthesisMemo.key(state.theta, state.log_surface_pressure)

            def _temperature(_g=g):
                exner = (_g["p_full"] / REFERENCE_PRESSURE_PA) ** KAPPA
                return _g["theta"] * exner

            g["temperature"] = self._band_value(
                "temperature", key, rows, _temperature
            )
        if "virtual_temperature" in need:
            key = SynthesisMemo.key(
                state.theta, state.log_surface_pressure, state.qv,
                *(getattr(state, name) for name in CONDENSATE_SPECIES),
            )

            def _virtual(_g=g):
                condensate = sum(
                    _g[name] for name in ("qc", "qr", "qi", "qs", "qg")
                )
                return _g["temperature"] * (
                    1.0 + 0.61 * _g["qv"] - condensate
                )

            g["virtual_temperature"] = self._band_value(
                "virtual_temperature", key, rows, _virtual
            )
        if need & {"u", "v"}:
            g["u"], g["v"] = self._band_value(
                "wind",
                SynthesisMemo.key(state.vorticity, state.divergence),
                rows,
                lambda: self._wind_band(sources.wind, rows),
            )
        if "geopotential" in need:
            key = SynthesisMemo.key(
                state.theta, state.log_surface_pressure, state.qv,
                *(getattr(state, name) for name in CONDENSATE_SPECIES),
            )

            def _geopotential(_g=g, _rows=rows):
                return self._hydrostatic_geopotential(
                    _g["virtual_temperature"], _g["p_half"], _g["p_full"],
                    _g["ln_ratio"], rows=_rows,
                )

            g["geopotential"] = self._band_value(
                "geopotential", key, rows, _geopotential
            )
        return g, stack, names

    def _mass_flux_sources(self, sources: GridSources):
        """The mass-flux divergence of a state, band-fed and band-drained.

        The analysis is fed ``dp*u`` and ``dp*v`` a band at a time into
        one full-latitude Fourier waist and contracted once; its grid
        divergence comes back as a second waist, and every consumer takes
        ``ps_t`` and the half-level pressure velocity from its own band,
        because both follow from that band's own columns (a level sum and
        a prefix sum over levels).
        """
        memo = self._memo
        state = sources.state
        key = SynthesisMemo.key(
            state.log_surface_pressure, state.vorticity, state.divergence
        )
        cached = memo.get("mass_flux", key) if memo is not None else None
        if cached is not None:
            return cached
        xp = self.transform.backend.xp
        waist = self.transform.open_waist(
            (2, self.nlev), bands=self.pipeline.bands
        )
        for rows in self.pipeline.local_slices():
            g, _stack, _names = self.grid_band(sources, rows)
            fluxes = xp.empty(
                (2, self.nlev, rows.stop - rows.start,
                 self.transform.grid.nlon),
                dtype=self.transform.backend.float_dtype,
            )
            xp.multiply(g["dp"], g["u"], out=fluxes[0])
            xp.multiply(g["dp"], g["v"], out=fluxes[1])
            del g
            waist.fill_band(rows.start, rows.stop, fluxes)
            del fluxes
        _zeta, divergence = self.vector._vordiv_from_fourier(waist.close())
        del waist
        spectral = self.transform.project(divergence)
        if self.pipeline.resident:
            grid = BandedField(
                self.transform, whole=self.transform.inverse(divergence)
            )
        else:
            grid = BandedField(
                self.transform,
                waist=self.transform.contract_to_waist(
                    divergence, bands=self.pipeline.bands
                ),
            )
        cached = (spectral, grid)
        if memo is not None:
            memo.put("mass_flux", key, cached, (
                state.log_surface_pressure, state.vorticity, state.divergence,
            ))
        return cached

    def _mass_flux_band(self, grid: BandedField, rows: slice):
        """``(div_mass, ps_t, omega_half)`` of one band.

        Every one of the three is that band's own columns: the level sum
        that closes the surface-pressure tendency and the prefix sum that
        makes the pressure velocity both run down a column.
        """
        xp = self.transform.backend.xp
        div_mass = grid.band(rows)
        ps_t = -xp.sum(div_mass, axis=0)
        omega_half, _closure = self.vertical.continuity(
            div_mass, ps_t, self.transform.backend
        )
        return div_mass, ps_t, omega_half

    def _mass_flux_and_omega(self, g_or_sources):
        """The whole-globe form, for callers outside the band loop."""
        sources = g_or_sources
        spectral, grid = self._mass_flux_sources(sources)
        div_mass, ps_t, omega_half = self._mass_flux_band(grid, self.whole_rows)
        return spectral, div_mass, ps_t, omega_half

    def release_syntheses(self) -> None:
        """Drop every memoized synthesis (before the physics suite call,
        the step's device peak, and whenever a caller wants the memory)."""
        if self._memo is not None:
            self._memo.clear()
        self._band_values.clear()
        self._band_tag = None

    @property
    def semi_lagrangian(self) -> bool:
        """Whether this model integrates with the semi-Lagrangian core."""
        return str(self.integrator).lower() in SEMILAG_INTEGRATORS

    def trajectory_state(self):
        """The second time level the semi-Lagrangian core carries, or
        None before its first step (semilag.state.TrajectoryState)."""
        return getattr(self, "_trajectory", None)

    def set_trajectory_state(self, trajectory) -> None:
        """Install the second time level (the step, and the restart)."""
        self._trajectory = trajectory

    def hold_pre_physics(self, atmosphere) -> None:
        """Keep the state as it stood BEFORE the first physics half.

        Only ``semilag.physics_coupling`` reads it, and only to take the
        difference that half produced.  It is a reference to a state the
        step holds to its end anyway, so it costs no memory; it is cleared
        by whoever takes it, so a coupling that asked for it and did not
        get it fails loudly instead of running the other arithmetic.
        """
        self._pre_physics = atmosphere

    def take_pre_physics(self):
        """The held pre-physics state, once; None if none was held."""
        held = getattr(self, "_pre_physics", None)
        self._pre_physics = None
        return held

    @staticmethod
    def _stack_row_index(names, wanted) -> slice:
        """The contiguous row slice of a synthesis stack holding ``wanted``.

        Refuses when the names are not consecutive rows in that order: a
        strided gather here would silently hand the reductions below a
        differently shaped operand, and the stack's row order is what
        keeps them on the pre-refactor shapes.
        """
        first = names.index(wanted[0])
        if tuple(names[first : first + len(wanted)]) != tuple(wanted):
            raise AssertionError(
                f"stack rows {names} do not hold {wanted} consecutively"
            )
        return slice(first, first + len(wanted))

    def _pressure_gradient_factor(self, ps, p_half):
        """``grad(ln p_k)`` per unit ``grad(ln ps)`` at the full levels.

        ``ln p_k = (ln p_{k-1/2} + ln p_{k+1/2}) / 2`` and each interface
        gradient is ``B grad ps``, so ``grad(ln p_k) = factor_k grad(ln
        ps)`` with ``factor_k = (B_{k-1/2} / p_{k-1/2} + B_{k+1/2} /
        p_{k+1/2}) ps / 2``, exact on the grid.  The momentum term ``R Tv_k
        grad(ln p_k)`` then balances the geopotential of
        ``_hydrostatic_geopotential`` pointwise for a column with Tv linear
        in ln p (DN-3), and the gradient of the spectral ``ln ps`` the
        state already carries replaces the former forward transform of
        ``ln p_full`` (one 3-D transform per stage, and a band-limited
        ``grad(ln p)`` that the grid product then aliased).
        """
        return 0.5 * (
            self._b_upper[:, None, None] / p_half[:-1]
            + self._b_lower[:, None, None] / p_half[1:]
        ) * ps[None]

    def _mass_flux_divergence(self, dp, u, v):
        _, divergence = self.vector.flux_curl_divergence(dp * u, dp * v)
        grid = self.transform.inverse(divergence)
        return self.transform.project(divergence), grid

    def _vertical_momentum_advection(self, field, omega_half, p_full):
        xp = self.transform.backend.xp
        # Full-level pressures are strictly increasing because
        # HybridCoordinate.pressure refuses any nonpositive layer thickness,
        # so these differences divide safely; a floor here would understate
        # the derivative in layers thinner than the floor.  Sliced instead
        # of looped so leading (component) axes batch and a GPU backend
        # launches a handful of kernels rather than one per level.
        omega_full = 0.5 * (omega_half[:-1] + omega_half[1:])
        derivative = xp.empty_like(field)
        derivative[..., 1:-1, :, :] = (
            field[..., 2:, :, :] - field[..., :-2, :, :]
        ) / (p_full[2:] - p_full[:-2])
        derivative[..., 0, :, :] = (
            field[..., 1, :, :] - field[..., 0, :, :]
        ) / (p_full[1] - p_full[0])
        derivative[..., -1, :, :] = (
            field[..., -1, :, :] - field[..., -2, :, :]
        ) / (p_full[-1] - p_full[-2])
        return omega_full * derivative

    def _vertical_scalar_flux_divergence(self, scalar, omega_half, p_full, p_half):
        """Monotone second-order (van Leer / MUSCL) vertical flux divergence.

        The interface value is reconstructed from the UPSTREAM layer with a
        van Leer (harmonic-mean) limited gradient in pressure, evaluated at
        the interface pressure, and then bounded by the two adjacent layer
        values so the flux is monotone on the non-uniform hybrid layers
        (the geometric-mean full level sits above the layer's arithmetic
        centre, so an unbounded MUSCL extrapolation can overshoot by up to
        4% where the ln-p spacing shrinks toward the surface).  The top and
        bottom layers have no second neighbour and keep a zero gradient
        (first order at their one interior interface).

        Replaces the first-order donor-cell flux, whose modified equation
        carried a vertical diffusivity ``|omega| dp / 2`` on theta and every
        tracer (measured: 1.311e-4 K/s error against 3.848e-5 centred on
        the smooth-profile test, 5.6 h e-fold for a one-layer feature at
        omega 0.5 Pa/s; audit 2026-09-01, DN-4).  The limiter removes that
        diffusivity (4.185e-5 K/s on the same test, L1 order 2.04/2.03 on a
        40->80->160 level ladder) but, being C0 in the state like donor
        cell, leaves the SSPRK3 temporal-order symptom where it was
        (theta 1.6-2.2 against 3 for divergence, re-measured after the
        change).  Vectorized over interfaces AND
        any leading (tracer) axes (a per-interface loop on a GPU backend
        measured 22.5 s/step at T63 against 13 s on one CPU core, launch
        latency, not compute); the numpy expressions are the specification
        for the fused cupy kernel, same association throughout.
        """
        xp = self.transform.backend.xp
        nlev = self.nlev
        if self.transform.backend.name == "cupy":
            out = xp.empty(
                scalar.shape, dtype=self.transform.backend.float_dtype
            )
            horiz = scalar.shape[-2] * scalar.shape[-1]
            vertical_flux_divergence_kernel(xp)(
                scalar, omega_half, p_full, p_half, nlev, horiz, out
            )
            return out
        # Van Leer limited gradient (per Pa) of the interior layers; the
        # boundary layers keep zero.
        gradient_above = (
            scalar[..., 1:-1, :, :] - scalar[..., :-2, :, :]
        ) / (p_full[1:-1] - p_full[:-2])
        gradient_below = (
            scalar[..., 2:, :, :] - scalar[..., 1:-1, :, :]
        ) / (p_full[2:] - p_full[1:-1])
        product = gradient_above * gradient_below
        monotone = product > 0.0
        denominator = xp.where(monotone, gradient_above + gradient_below, 1.0)
        gradient = xp.zeros_like(scalar)
        gradient[..., 1:-1, :, :] = xp.where(
            monotone, 2.0 * product / denominator, 0.0
        )
        # Interfaces j = 1..nlev-1 sit between layer j-1 (above) and j
        # (below); omega >= 0 is downward, so its upstream layer is above.
        interior = omega_half[1:nlev]
        above = scalar[..., :-1, :, :]
        below = scalar[..., 1:, :, :]
        face_from_above = above + gradient[..., :-1, :, :] * (
            p_half[1:nlev] - p_full[:-1]
        )
        face_from_below = below + gradient[..., 1:, :, :] * (
            p_half[1:nlev] - p_full[1:]
        )
        face = xp.where(interior >= 0.0, face_from_above, face_from_below)
        face = xp.minimum(
            xp.maximum(face, xp.minimum(above, below)), xp.maximum(above, below)
        )
        flux = xp.zeros(
            (*scalar.shape[:-3], nlev + 1, *scalar.shape[-2:]),
            dtype=self.transform.backend.float_dtype,
        )
        flux[..., 1:nlev, :, :] = interior * face
        return flux[..., 1:, :, :] - flux[..., :-1, :, :]

    def _scalar_flux_band(self, scalar, dp, u, v):
        """The two horizontal mass fluxes of a band's scalars, contiguous.

        The pair is written straight into the ``(2, ...)`` object the
        vector analysis Fourier-transforms and released as soon as its
        transform exists: the former separate fluxes plus their
        ``xp.stack`` copy were 2.29 + 2.29 GiB per six-field chunk at T533
        float32, and this chunk loop is where the step ran out of memory
        after physics.  Same arithmetic: the fused kernel writes the same
        expression into caller-supplied outputs, the numpy branch keeps
        ``(dp * scalar) * u`` / ``* v`` with the shared product associated
        first, and the analysis sees the identical contiguous ``(2, ...)``
        operand ``xp.stack([east, north])`` produced.
        """
        xp = self.transform.backend.xp
        fluxes = xp.empty(
            (2, *scalar.shape), dtype=self.transform.backend.float_dtype
        )
        if self.transform.backend.name == "cupy":
            tracer_flux_kernel(xp)(dp, scalar, u, v, fluxes[0], fluxes[1])
        else:
            mass = dp * scalar
            xp.multiply(mass, u, out=fluxes[0])
            xp.multiply(mass, v, out=fluxes[1])
            del mass
        return fluxes

    def _scalar_tendency_band(
        self, horizontal, scalar, dp, dp_t, omega_half, p_full, p_half
    ):
        """The scalar tendency of a band, once its horizontal flux
        divergence has come back through the whole contraction.

        The divisor is the true layer thickness: dp > 0 is guaranteed by
        HybridCoordinate.pressure, and the flux form conserves dp*scalar
        only when the same dp scales the fluxes and the division.
        """
        vertical = self._vertical_scalar_flux_divergence(
            scalar, omega_half, p_full, p_half
        )
        if self.transform.backend.name == "cupy":
            return scalar_tendency_kernel(self.transform.backend.xp)(
                horizontal, vertical, scalar, dp_t, dp
            )
        return (-horizontal - vertical - scalar * dp_t) / dp

    def _scalar_tendency(self, scalar, dp, dp_t, u, v, omega_half, p_full, p_half):
        """The whole-globe scalar tendency, for callers outside the band loop."""
        fluxes = self._scalar_flux_band(scalar, dp, u, v)
        fourier = self.vector._wind_fourier(fluxes)
        del fluxes
        _, horizontal_spec = self.vector._vordiv_from_fourier(fourier)
        del fourier
        horizontal = self.transform.inverse(horizontal_spec)
        del horizontal_spec
        return self._scalar_tendency_band(
            horizontal, scalar, dp, dp_t, omega_half, p_full, p_half
        )

    def rhs(self, state: MoistHybridState) -> MoistHybridState:
        prof = profiler_of(self)
        with prof.section("rhs"):
            return self._rhs(state, prof)

    def _rhs(self, state: MoistHybridState, prof) -> MoistHybridState:
        with prof.section("grid_state"):
            sources = self.grid_sources(state)
        return self._rhs_from_sources(state, sources, prof)

    def _rhs_from_sources(self, state, sources, prof) -> MoistHybridState:
        """The explicit right-hand side, one latitude band at a time.

        THREE PASSES OVER THE BANDS, and the count is the combine schedule,
        not a choice.  A partial spectral sum can be carried forward while
        everything downstream of it is linear in the coefficients; it has
        to be combined before it is synthesised back to the grid.  The
        mass-flux divergence is analysed and immediately inverted, so the
        bands that build its flux pair cannot be the bands that read its
        grid divergence; the scalar advection's horizontal flux divergence
        is analysed and immediately inverted for the same reason.  So:

        1. the mass-flux pass (:meth:`_mass_flux_sources`) fills one
           full-latitude Fourier waist with ``dp*u`` and ``dp*v`` and
           contracts it once;
        2. the momentum pass drains that divergence a band at a time,
           builds the two momentum forcings, the Bernoulli function, the
           surface-pressure tendency and the scalar flux pair on the
           band's own columns, and fills four more waists;
        3. the scalar pass drains the horizontal flux divergence and
           closes the tendency.

        Every contraction between them runs at the shape a resident stage
        gives it, because a waist is whole in latitude however many bands
        filled it.
        """
        xp = self.transform.backend.xp
        pipeline = self.pipeline
        nlon = self.transform.grid.nlon
        limit = max(1, int(self.spectral_chunk))
        with prof.section("momentum"):
            _spectral, div_mass = self._mass_flux_sources(sources)
            # The ln ps gradient is two PLANES (5.13 MB each at T533), so
            # it is taken whole once and the band reads its own rows.
            grad_lnps_east, grad_lnps_north = self.transform.gradient(
                state.log_surface_pressure
            )
            momentum_waist = self.transform.open_waist(
                (2, self.nlev), bands=pipeline.bands
            )
            bernoulli_waist = self.transform.open_waist(
                (self.nlev,), bands=pipeline.bands
            )
            logps_t_waist = self.transform.open_waist((), bands=pipeline.bands)
            scalars_rows = None
            flux_waists = []
            for rows in pipeline.local_slices():
                g, stack, names = self.grid_band(sources, rows)
                if scalars_rows is None:
                    scalars_rows = self._stack_row_index(names, ("theta", "qv"))
                _dm, ps_t, omega_half = self._mass_flux_band(div_mass, rows)
                del _dm
                # ps = exp(logps) is positive for every finite spectral
                # state, and enforce() refuses the run outside
                # 30000..120000 Pa, so the true surface pressure divides
                # safely.
                logps_t_waist.fill_band(rows.start, rows.stop, ps_t / g["ps"])
                dp_t = self._delta_b[:, None, None] * ps_t[None]
                del ps_t
                w_uv = self._vertical_momentum_advection(
                    xp.stack([g["u"], g["v"]]), omega_half, g["p_full"]
                )
                pressure_gradient = self._pressure_gradient_factor(
                    g["ps"], g["p_half"]
                )
                if self.transform.backend.name == "cupy":
                    # One fused launch produces both momentum forcings and
                    # the Bernoulli function; the numpy expressions below
                    # are the specification (same association, only fma
                    # contraction moves last bits).
                    momentum_u, momentum_v, bernoulli = (
                        momentum_bernoulli_kernel(xp)(
                            g["vorticity"],
                            self.coriolis[rows],
                            g["u"], g["v"], w_uv[0], w_uv[1],
                            g["virtual_temperature"],
                            pressure_gradient,
                            grad_lnps_east[rows], grad_lnps_north[rows],
                            g["geopotential"],
                            self.transform.backend.float_dtype(self.gas_constant),
                        )
                    )
                else:
                    absolute_vorticity = g["vorticity"] + self.coriolis[rows]
                    pressure_force = (
                        self.gas_constant * g["virtual_temperature"]
                        * pressure_gradient
                    )
                    momentum_u = (
                        absolute_vorticity * g["v"]
                        - w_uv[0]
                        - pressure_force * grad_lnps_east[rows]
                    )
                    momentum_v = (
                        -absolute_vorticity * g["u"]
                        - w_uv[1]
                        - pressure_force * grad_lnps_north[rows]
                    )
                    bernoulli = (
                        0.5 * (g["u"] ** 2 + g["v"] ** 2) + g["geopotential"]
                    )
                del w_uv, pressure_gradient
                pair = xp.empty(
                    (2, *momentum_u.shape),
                    dtype=self.transform.backend.float_dtype,
                )
                pair[0] = momentum_u
                pair[1] = momentum_v
                del momentum_u, momentum_v
                momentum_waist.fill_band(rows.start, rows.stop, pair)
                del pair
                bernoulli_waist.fill_band(rows.start, rows.stop, bernoulli)
                del bernoulli
                # theta and every tracer stack into one advection call and
                # one analysis: same batching argument as the momentum
                # block.  The stack is the synthesis stack's own leading
                # rows (no copy).
                scalars = stack[scalars_rows]
                for index, start in enumerate(
                    range(0, scalars.shape[0], limit)
                ):
                    chunk = scalars[start : start + limit]
                    if len(flux_waists) <= index:
                        flux_waists.append((start, self.transform.open_waist(
                            (2, chunk.shape[0], self.nlev),
                            bands=pipeline.bands,
                        )))
                    flux_waists[index][1].fill_band(
                        rows.start, rows.stop,
                        self._scalar_flux_band(
                            chunk, g["dp"], g["u"], g["v"]
                        ),
                    )
                    del chunk
                del scalars, stack, g, omega_half, dp_t
            del grad_lnps_east, grad_lnps_north
            zeta_t, divergence_t = self.vector._vordiv_from_fourier(
                momentum_waist.close()
            )
            del momentum_waist
            divergence_t = divergence_t - self.transform.laplacian(
                self.transform.contract_waist(bernoulli_waist.close())
            )
            del bernoulli_waist
            logps_spectral = self.transform.project(
                self.transform.contract_waist(logps_t_waist.close())
            )
            del logps_t_waist
            horizontal = []
            for start, waist in flux_waists:
                _, horizontal_spec = self.vector._vordiv_from_fourier(
                    waist.close()
                )
                if pipeline.resident:
                    horizontal.append((start, BandedField(
                        self.transform,
                        whole=self.transform.inverse(horizontal_spec),
                    )))
                else:
                    horizontal.append((start, BandedField(
                        self.transform,
                        waist=self.transform.contract_to_waist(
                            horizontal_spec, bands=pipeline.bands
                        ),
                    )))
                del horizontal_spec
            del flux_waists

        with prof.section("scalars"):
            # Each chunk's tendency is analysed as soon as it exists
            # instead of after every chunk's grid tendency has been
            # assembled: the chunk handed to the analysis is the same
            # field stack either way, so the coefficients are the same
            # bits.
            tendency_waists = [
                (start, self.transform.open_waist(
                    (field.waist.lead[0] if field.waist is not None
                     else field.whole.shape[0], self.nlev),
                    bands=pipeline.bands,
                ))
                for start, field in horizontal
            ]
            for rows in pipeline.local_slices():
                g, stack, names = self.grid_band(sources, rows)
                _dm, ps_t, omega_half = self._mass_flux_band(div_mass, rows)
                del _dm
                dp_t = self._delta_b[:, None, None] * ps_t[None]
                del ps_t
                scalars = stack[scalars_rows]
                for (start, field), (_start, waist) in zip(
                    horizontal, tendency_waists
                ):
                    chunk = scalars[start : start + limit]
                    waist.fill_band(rows.start, rows.stop, (
                        self._scalar_tendency_band(
                            field.band(rows), chunk, g["dp"], dp_t,
                            omega_half, g["p_full"], g["p_half"],
                        )
                    ))
                    del chunk
                del scalars, stack, g, omega_half, dp_t
            del horizontal, div_mass
            scalar_t_spectral = None
            for start, waist in tendency_waists:
                piece = self.transform.contract_waist(waist.close())
                if scalar_t_spectral is None and len(tendency_waists) == 1:
                    scalar_t_spectral = piece
                    break
                if scalar_t_spectral is None:
                    scalar_t_spectral = xp.empty(
                        (2, *piece.shape[1:]), dtype=piece.dtype
                    )
                scalar_t_spectral[start : start + piece.shape[0]] = piece
                del piece
            del tendency_waists
            scalar_t_spectral = self.transform.project(scalar_t_spectral)
            divergence_spectral = self.transform.project(divergence_t)
        with prof.section("linear"):
            # The semi-implicit split: the linear gravity-wave operator is
            # REMOVED from what the explicit integrator sees and integrated by
            # Crank-Nicolson in the scheme's pre/post maps (integrate_dynamics).
            # Leaving it here while those also integrate it double-steps the
            # wave - the defect that amplified surface pressure to 1256 hPa by
            # hour 9 of the first real-data T63 run.  The vertical-mode
            # operator also carries a theta tendency; the barotropic proxy
            # does not.
            theta_spectral = scalar_t_spectral[0]
            linear = self.semi_implicit.linear_tendencies(
                state, self.transform, self.vertical
            )
            if linear is not None:
                divergence_spectral = self.transform.project(
                    divergence_spectral - linear.divergence
                )
                if linear.theta is not None:
                    theta_spectral = self.transform.project(
                        theta_spectral - linear.theta
                    )
                logps_spectral = self.transform.project(
                    logps_spectral - linear.log_surface_pressure
                )
        # The tendency carries no grid-tracer rows: the grid tracers are
        # advanced once per step by the flux-form transport
        # (_transport_grid_tracers), not by the explicit integrator.
        return MoistHybridState(
            vorticity=self.transform.project(zeta_t),
            divergence=divergence_spectral,
            theta=theta_spectral,
            log_surface_pressure=logps_spectral,
            qv=scalar_t_spectral[1],
            time_s=state.time_s,
            step=state.step,
        )

    def cfl(self, state: MoistHybridState, dt_s: float) -> float:
        # Only the wind bounds the advective CFL, so only the wind is
        # synthesized - a full grid_state here paid fifteen fields for two.
        xp = self.transform.backend.xp
        u, v = self._wind(state)
        maximum = float(self.transform.backend.to_numpy(
            _associative_over(
                xp, "max", xp.sqrt(u * u + v * v), name="cfl_wind",
                exchange=self.pipeline.exchange,
            )
        ))
        # The gravity wave counts against the explicit budget only for the
        # fraction the semi-implicit split leaves explicit (its fastest
        # treated mode times 1 - weight).  Under the external-only proxy
        # this gate never bounded the internal modes it left explicit
        # (audit 2026-09-01, DN-1: dt=60 s unstable at 142.7 m/s while the
        # gate tripped at 149.3).  The vertical-mode default treats every
        # mode, but at the shipped off-centring alpha = 0.5 the gate is not
        # a strict bound either: the explicit residual of the split
        # Doppler-shifted by the wind grows at rho up to 1.0007 per step at
        # dt = 60 s, U = 150 m/s (gate 149.3 m/s) and 1.0015 at dt = 120 s,
        # U = 50 m/s (gate 74.6), measured on the 20-level pressure_blend
        # stack at T533 as the largest spectral radius over degrees.  That
        # is 100-1000x slower than the proxy's growth past its gate and an
        # e-folding time of about 25 days at the shipped dt = 30 s;
        # alpha = 0.55 removes it (rho = 1 at every U and dt of that scan).
        characteristic = maximum + self.semi_implicit.explicit_wave_speed_m_s(
            self.vertical
        )
        return float(dt_s) * characteristic * math.sqrt(
            self.transform.truncation * (self.transform.truncation + 1.0)
        ) / self.transform.grid.radius_m

    def implicit_wave_number(self, dt_s: float) -> float:
        """``c_max k_T dt``: the fastest treated mode's omega dt at the
        truncation, the argument of the IMEX integrator's implicit
        stability function (zero under the split-era steppers)."""
        integrator = str(self.integrator).lower()
        if integrator not in IMEX_TABLEAUX or not getattr(self.semi_implicit, "active", False):
            return 0.0
        if isinstance(self.semi_implicit, VerticalModeSemiImplicit):
            fastest = self.semi_implicit.operator(self.vertical).fastest_mode_m_s
        else:
            fastest = self.semi_implicit.external_wave_speed_m_s
        k_top = math.sqrt(
            self.transform.truncation * (self.transform.truncation + 1.0)
        ) / self.transform.grid.radius_m
        return float(dt_s) * fastest * k_top * self.semi_implicit.divergence_weight

    def _refuse_beyond_implicit_neutral_limit(self, dt_s: float) -> None:
        """The IMEX member's implicit stability function is neutral or
        damping only up to omega dt = IMPLICIT_NEUTRAL_LIMIT (17.3 for the
        shipped tableau); beyond it the stage pair AMPLIFIES the fastest
        treated modes at the truncation on its own (|R(i y)| ~ y / 15 for
        large y), the linearized rest ceiling measured 583 s at T533 on
        the 40-level default with the shipped hyperdiffusion.  A step past
        the limit is refused by name instead of growing at the truncation
        for hours."""
        integrator = str(self.integrator).lower()
        if integrator in SEMILAG_INTEGRATORS:
            # Named rather than skipped (gate law).  The off-centred
            # two-time-level map's amplification factor is
            # |(1 + i(1-alpha) y)/(1 - i alpha y)|, which is <= 1 for
            # EVERY y when alpha >= 0.5, and alpha < 0.5 is already
            # refused by VerticalModeSemiImplicit.__post_init__.  So this
            # path has no wave-number ceiling at all and the binding
            # constraint is the trajectory, which the Lipschitz gate in
            # step() reads.  Returning here without saying so would make
            # a future integrator inherit a silent skip.
            return
        limit = IMPLICIT_NEUTRAL_LIMIT.get(integrator)
        if limit is None:
            return
        y = self.implicit_wave_number(dt_s)
        if y > limit:
            raise ValueError(
                f"WOOF global {integrator} implicit stage pair amplifies the "
                f"fastest gravity-wave mode at the truncation beyond omega dt = "
                f"{limit:.1f}: this step has {y:.1f} (c_max k_T dt); reduce dt_s "
                "or truncation"
            )

    def integrate_dynamics(self, atmosphere: MoistHybridState, dt_s: float, mark=None):
        """The adiabatic core over one step.

        Under an IMEX integrator (imex.py; ``integrator`` names a tableau
        in ``IMEX_TABLEAUX``) the explicit right-hand side (``rhs``, which
        carries ``rhs - L``) and the scheme's operator ``L`` advance
        together in one Runge-Kutta step whose two tableaux share every
        abscissa, so a balanced state is an exact fixed point.  Under
        ``ssprk3`` / ``rk4`` the split of the earlier era runs unchanged:
        the scheme's pre-map, the explicit integrator, the scheme's
        post-map (the vertical-mode scheme a symmetric split, sqrt of
        Crank-Nicolson on either side; the external proxy the Lie split
        of its era).  Returns the advanced atmosphere and the
        semi-implicit metric (the largest divergence increment the
        implicit part made).  ``mark(name, atmosphere)``, when given, is
        the energy ledger's observer called around each operator of the
        split (insitu.energy).  The energy-tendency harness calls this to
        reproduce exactly the dynamics stage of :meth:`step`.
        """
        key = "semi_implicit_max_divergence_increment_s1"
        if str(self.integrator).lower() in SEMILAG_INTEGRATORS:
            return semilag_step(self, atmosphere, float(dt_s), mark=mark)
        tableau = IMEX_TABLEAUX.get(str(self.integrator).lower())
        if tableau is not None:
            advanced, metrics = imex_step(
                atmosphere, float(dt_s), self.rhs, self.semi_implicit,
                self.transform, self.vertical, tableau, mark=mark,
                profiler=profiler_of(self),
            )
            if mark is not None:
                # imex_step marked y_n plus the explicit sum alone as
                # dynamics_explicit; this mark's net is the implicit
                # operator's sum (insitu.energy, 2026-09-04).
                mark("dynamics_implicit", advanced)
            return advanced, {key: float(metrics[key])}
        pre, before = self.semi_implicit.pre_apply(
            atmosphere, self.transform, self.vertical, dt_s
        )
        if mark is not None:
            mark("semi_implicit_pre", pre)
        advanced = step_with_scheme(pre, float(dt_s), self.rhs, self.integrator)
        if mark is not None:
            mark("explicit_dynamics", advanced)
        advanced, after = self.semi_implicit.post_apply(
            advanced, self.transform, self.vertical, dt_s
        )
        if mark is not None:
            mark("semi_implicit_post", advanced)
        return advanced, {key: max(float(before[key]), float(after[key]))}

    def _apply_eddy_viscosity(self, state: MoistHybridState, dt_s: float):
        """The spectral eddy viscosity (woof.globe.spectral.eddy_viscosity):
        every level reads its own kinetic energy at the truncation from the
        vorticity and divergence coefficients already in hand, the
        velocity fields take nu_e(n), the scalars nu_e(n) / Pr_t, and the
        log surface pressure the scalar drain of the lowest level.  The
        last reading is kept on ``closure_record`` for the receipt."""
        from .insitu.spectra import SpectralKineticEnergy

        closure: SpectralEddyViscosity = self.diffusion
        ke = getattr(self, "_closure_spectral_ke", None)
        if ke is None:
            ke = SpectralKineticEnergy(self.transform)
            object.__setattr__(self, "_closure_spectral_ke", ke)
        to_numpy = self.transform.backend.to_numpy
        rot = ke.by_degree(state.vorticity)
        div = ke.by_degree(state.divergence)
        by_degree = np.asarray(to_numpy(rot + div), dtype=np.float64)
        nu_inf = closure.nu_infinity(by_degree, self.transform)
        bottom = int(np.argmax(self._closure_reference_pressures()))
        momentum = closure.factors(self.transform, dt_s, nu_inf)
        scalar = momentum ** (1.0 / float(closure.eddy_prandtl))
        project = self.transform.project
        fields = []
        for name in SPECTRAL_FIELDS:
            coeff = getattr(state, name)
            if name in ("vorticity", "divergence"):
                fields.append(project(coeff * momentum[..., None]))
            elif name == "log_surface_pressure":
                fields.append(project(coeff * scalar[bottom][:, None]))
            else:
                fields.append(project(coeff * scalar[..., None]))
        object.__setattr__(self, "closure_record", {
            "closure": "spectral_eddy_viscosity",
            "nu_infinity_m2_s": [float(v) for v in nu_inf],
            "energy_at_truncation_m2_s2": [float(v) for v in by_degree[..., -1]],
            "bottom_level": bottom,
            "dt_s": float(dt_s),
        })
        return state.with_fields(fields)

    def _closure_reference_pressures(self) -> np.ndarray:
        """Full-level reference pressures at a 1000 hPa surface, to name
        the lowest level whatever the index order of the ladder."""
        a = np.asarray(self.vertical.a_half_pa, dtype=np.float64)
        b = np.asarray(self.vertical.b_half, dtype=np.float64)
        p_half = a + b * 1.0e5
        return np.sqrt(np.maximum(p_half[:-1], 1.0) * p_half[1:])

    def _apply_diffusion(self, state: MoistHybridState, dt_s: float):
        if self.diffusion is None:
            return state
        if isinstance(self.diffusion, SpectralEddyViscosity):
            return self._apply_eddy_viscosity(state, dt_s)
        fields = []
        for name in SPECTRAL_FIELDS:
            strength = 1.0
            if name == "divergence":
                strength = self.divergence_diffusion_strength
            elif name == "log_surface_pressure":
                strength = self.pressure_diffusion_strength
            elif name == "qv":
                strength = self.water_diffusion_strength
            fields.append(
                self.diffusion.apply(
                    getattr(state, name), self.transform, dt_s, strength=strength
                )
            )
        return state.with_fields(fields)

    def initialize_mass_target(self, state: MoistHybridState) -> float:
        ps = self.transform.backend.to_numpy(
            self.transform.backend.xp.exp(
                self.transform.inverse(state.log_surface_pressure)
            )
        )
        self._target_mass_pa = self.transform.grid.global_mean(ps)
        return float(self._target_mass_pa)

    def _fix_mass(self, state: MoistHybridState):
        if not self.mass_fixer:
            return state, 0.0
        if self._target_mass_pa is None:
            self.initialize_mass_target(state)
        xp = self.transform.backend.xp
        logps = self.transform.inverse(state.log_surface_pressure)
        # The surface-pressure plane assembles whole into a resident
        # buffer (5.13 MB at T533) and then takes the same host global
        # mean, which is itself already two-stage (a zonal mean per row,
        # then one weighted sum over latitude in grid order).
        plane = PlaneAccumulator(
            xp, logps.shape, self.transform.backend.float_dtype,
            name="fix_mass_surface_pressure", exchange=self.pipeline.exchange,
        )
        plane.add_band(slice(0, logps.shape[-2]), xp.exp(logps))
        current = self.transform.grid.global_mean(
            self.transform.backend.to_numpy(plane.plane)
        )
        if not math.isfinite(current) or current <= 0.0:
            raise FloatingPointError(f"invalid global mean surface pressure {current!r}")
        correction = math.log(float(self._target_mass_pa) / current)
        fixed = self.transform.add_grid_constant(
            state.log_surface_pressure, correction
        )
        values = list(state.fields())
        values[3] = fixed
        return state.with_fields(values), correction

    def _level_water_mass_rows(self, stacked, dp):
        """The band-local stage of :meth:`_level_water_mass`: the zonal
        mean of each (species, level, latitude row).  Every row is
        independent, so a band computes exactly the rows it holds."""
        xp = self.transform.backend.xp
        return xp.mean(stacked * dp, axis=-1)

    def _level_water_mass_total(self, accumulator):
        """The whole stage: one weighted sum over latitude, in grid order."""
        weights = self.transform.backend.asarray(
            self.transform.grid.quadrature_weights,
            dtype=self.transform.backend.float_dtype,
        )
        return 0.5 * accumulator.total(weights) / GRAVITY_M_S2

    def _level_water_mass(self, stacked, dp):
        """Area-weighted global mean of each (species, level) water mass,
        kg/m2: ``stacked`` is (n_species, nlev, nlat, nlon) mixing ratio
        and ``dp`` the layer thickness in Pa.  Summed over species and
        levels this is the global-mean atmospheric water column.

        Two stages over a resident (species, level, latitude) buffer (256
        KB at T533), so a run that presents the rows one band at a time
        reduces the same operand in the same grid order and gets the same
        number.  The resident path below is that buffer filled by a single
        band covering every row.
        """
        xp = self.transform.backend.xp
        rows = self._level_water_mass_rows(stacked, dp)
        accumulator = LatitudeAccumulator(
            xp, rows.shape, rows.dtype, name="level_water_mass", exchange=self.pipeline.exchange,
        )
        accumulator.add_band(slice(0, rows.shape[-1]), rows)
        return self._level_water_mass_total(accumulator)

    def _column_hole_accumulators(self):
        """The column-hole filler's three resident readings.

        The two mass readings are flat sums over a horizontal plane, and a
        flat sum cannot be folded band by band without changing its last
        bits, so the PLANE is the accumulator (5.13 MB at T533) and the
        flat sum runs once over it.  The rescale reading is a maximum,
        exact in any order.
        """
        xp = self.transform.backend.xp
        shape = self.transform.grid.shape
        dtype = self.transform.backend.float_dtype
        return {
            "created": PlaneAccumulator(
                xp, shape, dtype, name="column_holes_created", exchange=self.pipeline.exchange),
            "unfillable": PlaneAccumulator(
                xp, shape, dtype, name="column_holes_unfillable", exchange=self.pipeline.exchange),
            "rescale": AssociativeAccumulator(
                xp, "max", name="column_holes_rescale", exchange=self.pipeline.exchange),
        }

    def _fill_column_holes_band(self, q, dp, accumulators, rows: slice):
        """One band of :meth:`_fill_column_holes`.

        Every expression here is column-local: the two level sums, the
        fillable test, the per-column scale and the rescaled field all
        read one column and write it, so a band's rows are the rows the
        whole call would have written there.  Only the three readings
        cross a band edge, and they cross it through the accumulators.
        """
        xp = self.transform.backend.xp
        weighted = q * dp
        negative = xp.sum(xp.minimum(weighted, 0.0), axis=0)
        positive = xp.sum(xp.maximum(weighted, 0.0), axis=0)
        del weighted
        fillable = (positive > 0.0) & (positive + negative > 0.0)
        scale = xp.where(
            fillable,
            (positive + negative) / xp.where(fillable, positive, 1.0),
            1.0,
        )
        del positive
        filled = xp.maximum(q, 0.0) * scale[None]
        accumulators["created"].add_band(rows, negative)
        accumulators["unfillable"].add_band(
            rows, xp.where(fillable, 0.0, negative)
        )
        accumulators["rescale"].add_band(1.0 - scale)
        return filled

    def _fill_column_holes_readings(self, accumulators):
        """``(created_kg_m2, max_column_fraction, unfillable_kg_m2)``.

        The three readings cross to the host in ONE read (three
        synchronisations per call, twelve per step, before).
        """
        xp = self.transform.backend.xp
        cell = self.transform.backend.asarray(
            self.transform.grid.quadrature_weights,
            dtype=self.transform.backend.float_dtype,
        )[:, None] / (2.0 * self.transform.grid.nlon)
        readings = self.transform.backend.to_numpy(xp.stack([
            -accumulators["created"].total(cell),
            -accumulators["unfillable"].total(cell),
            accumulators["rescale"].total(),
        ]))
        return (
            max(0.0, float(readings[0]) / GRAVITY_M_S2),
            float(readings[2]),
            max(0.0, float(readings[1]) / GRAVITY_M_S2),
        )

    def _fill_column_holes(self, q, dp):
        """Column-local, conservative hole filling of a spectral water field.

        ``q`` is the synthesized (nlev, nlat, nlon) mixing ratio of a
        field that lives in the spectral basis (water vapor), ``dp`` the
        layer thickness.  Clipping the negative ringing of a truncated
        field ADDS mass; this filler removes exactly that mass again from
        the positive part of the SAME COLUMN, so every column's water
        integral is unchanged by the clip and nothing moves between
        columns.  Returns ``(filled, created_kg_m2, max_column_fraction,
        unfillable_kg_m2)``: ``created_kg_m2`` is the global-mean column
        water the clip created (the fixer's magnitude), ``max_column_
        fraction`` the largest fraction any column lost of its own
        positive water to fill its holes, and ``unfillable_kg_m2`` the
        part of ``created`` that stayed created because its column had no
        positive water to pay it (net-negative columns keep their plain
        clip: wiping them would be silent data destruction, and the
        receipt's positivity-fixer gate is the instrument that names such
        a state).

        Why column-local and not the per-level global rescale it
        replaces (v5, retired 2026-09-02): the v5 fixer multiplied EVERY
        column's water on a level by ``mass_before / mass_after`` of the
        whole level, which moved the mass the clip created in the
        negative lobes of one feature out of every cloud and rain shaft on
        the planet into clear air, where the microphysics evaporated it
        (37 percent of the world's cloud water per repair pass, four
        passes per step).  That fee belonged to the truncated
        representation of the condensate fields, which no longer have one
        (they are grid tracers); the vapor field's ringing is 0.00
        percent of its mass, and a column that rings pays its own lobes
        from its own vapor.  Why not the reservoir: v4 paid the created
        water out of the surface reservoir as a uniform levy, a
        reservoir-to-atmosphere moisture source of 2.508 mm/day; v3
        debited the ring column's reservoir and drained one at a
        persistently convective column in 37.7 h.  No reservoir transfer
        of any kind remains.

        This is the WHOLE-GLOBE call: one band covering every row.
        """
        accumulators = self._column_hole_accumulators()
        filled = self._fill_column_holes_band(
            q, dp, accumulators, self.whole_rows
        )
        created, max_fraction, unfillable = self._fill_column_holes_readings(
            accumulators
        )
        return filled, created, max_fraction, unfillable

    # -- the pinned host tier ------------------------------------------

    def _spills(self, slice_name: str) -> bool:
        return self.host_tier is not None and slice_name in self.spill_slices

    def _park_tracers(self, tracers):
        """Return the grid tracers to the tier, if it holds them."""
        if not self._spills("tracers"):
            return tracers
        return self.host_tier.hold("tracer", tracers)

    def _park_surface(self, surface):
        """Return the surface reservoirs to the tier, if it holds them."""
        if not self._spills("surface"):
            return surface
        from .state import SURFACE_ARRAY_NAMES, SurfaceState

        held = self.host_tier.hold("surface", {
            member: getattr(surface, member) for member in SURFACE_ARRAY_NAMES
        })
        return SurfaceState(**held)

    def _park_physics_state(self, physics_state):
        """Return the native physics namespace to the tier, if it holds it."""
        if not self._spills("physics"):
            return physics_state
        from .state import PhysicsState

        return PhysicsState(
            schema=physics_state.schema,
            arrays=self.host_tier.hold("physics", physics_state.arrays),
            metadata=physics_state.metadata,
        )

    def park_persistent(self, bundle: ArwenGlobalState) -> ArwenGlobalState:
        """The whole persistent grid state into the tier, once.

        Called at run start on the state the run begins from (a cold
        start or a restart): every later step returns its own slices
        through the three helpers above, into the slots this call made.
        """
        if self.host_tier is None or not self.spill_slices:
            return bundle
        atmosphere = bundle.atmosphere
        if self._spills("tracers"):
            atmosphere = atmosphere.with_grid_tracers(
                self._park_tracers(atmosphere.grid_tracers())
            )
        return ArwenGlobalState(
            atmosphere,
            self._park_surface(bundle.surface),
            self._park_physics_state(bundle.physics_state),
        )

    def _floor_grid_tracers(self, tracers):
        """Floor the grid tracers at zero and measure what the floor took.

        The grid tracers are nonnegative by construction (the transport
        is positive-definite and every physics suite floors its own
        result), so the floor is a roundoff instrument, not a fixer: it
        returns ``(floored, negative_kg_m2, largest_negative)`` where
        ``negative_kg_m2`` is the global-mean column mass of everything
        below zero across the condensate species and ``largest_negative``
        the most negative mixing ratio or number seen.  A floor that took
        more than the working dtype's roundoff of a field's own maximum
        refuses: a negative grid tracer beyond roundoff is a transport or
        physics defect, not ringing.
        """
        xp = self.transform.backend.xp
        host = self.transform.backend.to_numpy
        eps = float(xp.finfo(self.transform.backend.float_dtype).eps)
        out = {}
        largest = 0.0
        # Every tracer's minimum is reduced where it lives and the ten
        # scalars cross to the host in one read: the per-tracer read was
        # ten synchronisations per call, sixty per step (profile
        # 2026-09-04).  Same values, same order of checks.
        names = list(tracers)

        def _refuse(name, minimum, maximum):
            if -minimum > 64.0 * eps * max(maximum, 1.0e-30):
                raise FloatingPointError(
                    f"grid tracer {name} is negative beyond roundoff "
                    f"(min {minimum:.3g} against max {maximum:.3g}): a "
                    "grid-point tracer is nonnegative by construction, "
                    "so this is a transport or physics defect, not "
                    "representation ringing"
                )

        if any(spilled(tracers[name]) for name in names):
            # THE TIER HOLDS THEM.  Staging all ten to read ten minima in
            # one host read would put the whole tracer slice back on the
            # card, which is exactly what the tier exists to prevent, so
            # the fold runs one tracer at a time and the ten host reads
            # are the price of the 1.9 GiB (T533) not being there.  Same
            # arithmetic: a minimum is a minimum of the same bytes, and
            # the floor writes the same values back to the same slot.
            for name in names:
                parked = tracers[name]
                value = resident(xp, parked)
                # The fold crosses the wire exactly as the resident path's
                # does: a card whose tier holds the tracers folded alone
                # while its peer, resident, waited on the collective, and
                # the pair deadlocked at step 1 (T383, RTX 5070 Ti with the
                # tier against an RTX 5090 without it, 2026-09-07).  Where
                # the arrays live is per card; the collectives a step
                # issues are not.
                minimum = float(host(
                    _associative_over(
                        xp, "min", value, name=f"floor_{name}",
                        exchange=self.pipeline.exchange,
                    )
                ))
                if minimum < 0.0:
                    maximum = float(host(_associative_over(
                        xp, "max", value, name=f"floor_{name}_max",
                        exchange=self.pipeline.exchange,
                    )))
                    _refuse(name, minimum, maximum)
                    largest = max(largest, -minimum)
                    # The floored field is HANDED BACK, not written into
                    # the tier's slot.  The resident path leaves the
                    # caller's state holding the unfloored array and only
                    # the returned dict floored, and one caller depends on
                    # exactly that: the physics exchange floors the tracers
                    # for the suite while its own band loop reads the
                    # state's arrays for the virtual temperature.  Writing
                    # the slot would hand that loop a different field from
                    # the one the resident run gives it.
                    out[name] = xp.maximum(value, 0.0)
                    del value
                    continue
                del value
                out[name] = parked
            return out, largest
        # Minimum and maximum are exactly associative in floating point, so
        # a band folds into them in any order and the answer is the whole
        # grid's either way.
        minima = host(xp.stack([
            _associative_over(
                xp, "min", tracers[name], name=f"floor_{name}",
                exchange=self.pipeline.exchange,
            )
            for name in names
        ]))
        for name, minimum in zip(names, minima):
            value = tracers[name]
            minimum = float(minimum)
            if minimum < 0.0:
                maximum = float(host(
                    _associative_over(
                        xp, "max", value, name=f"floor_{name}_max",
                        exchange=self.pipeline.exchange,
                    )
                ))
                _refuse(name, minimum, maximum)
                largest = max(largest, -minimum)
                value = xp.maximum(value, 0.0)
            out[name] = value
        return out, largest

    def _repair_positivity(self, bundle: ArwenGlobalState):
        """Clip the vapor's spectral ringing and close it inside its column.

        Only water vapor lives in the spectral basis; the grid tracers
        are nonnegative by construction and pass through untouched (a
        roundoff floor measures them).  Returns ``(bundle, |min vapor|,
        |min grid tracer|, fixer)`` where ``fixer`` is this call's
        record: ``water_kg_m2`` (global-mean column water the clip
        created and the column filler removed again), ``max_rescale``
        (the largest fraction any column lost of its own vapor to fill
        its holes), ``unfillable_kg_m2`` (created water in net-negative
        columns, left created) and ``atmospheric_water_kg_m2`` (the
        global-mean atmospheric water column, the denominator of the
        receipt's relative gate).
        """
        xp = self.transform.backend.xp
        host = self.transform.backend.to_numpy
        # A state without its grid tracers is refused by name here, before
        # the floor reads them (validate_state carries the sentence).
        self.validate_state(bundle.atmosphere)
        tracers, largest_negative_tracer = self._floor_grid_tracers(
            bundle.atmosphere.grid_tracers()
        )
        if not self.positivity_repair:
            atmosphere = bundle.atmosphere.with_grid_tracers(
                self._park_tracers(tracers)
            )
            return ArwenGlobalState(
                atmosphere, bundle.surface.copy(), bundle.physics_state.copy()
            ), 0.0, largest_negative_tracer, {
                "water_kg_m2": 0.0, "max_rescale": 0.0,
                "unfillable_kg_m2": 0.0, "atmospheric_water_kg_m2": 0.0,
            }
        # ONE BAND LOOP for the whole repair.  The vapor minimum is a
        # maximum's mirror and folds in any order; the filler's three
        # readings go into the resident planes above; the level water mass
        # goes into the resident (species, level, latitude) buffer; and
        # the filled field is fed straight into the analysis waist, so the
        # clipped vapor volume never exists at full size.
        sources = self.grid_sources(bundle.atmosphere, only=("qv", "dp"))
        holes = self._column_hole_accumulators()
        minimum = AssociativeAccumulator(xp, "min", name="repair_vapor_minimum", exchange=self.pipeline.exchange)
        water_rows = None
        waist = self.transform.open_waist(
            (self.nlev,), bands=self.pipeline.bands
        )
        for rows in self.pipeline.local_slices():
            g, _stack, _names = self.grid_band(sources, rows)
            qv = g["qv"]
            dp = g["dp"]
            minimum.add_band(qv)
            filled = self._fill_column_holes_band(qv, dp, holes, rows)
            del qv
            condensate = sum(
                band_view(xp, tracers[name], rows) for name in CONDENSATE_SPECIES
            )
            band_rows = self._level_water_mass_rows(
                xp.stack([filled, condensate]), dp
            )
            del condensate
            if water_rows is None:
                water_rows = LatitudeAccumulator(
                    xp, (*band_rows.shape[:-1], self.transform.grid.nlat),
                    band_rows.dtype, name="level_water_mass", exchange=self.pipeline.exchange,
                )
            water_rows.add_band(rows, band_rows)
            del band_rows
            waist.fill_band(rows.start, rows.stop, filled)
            del filled, g, dp
        del sources
        negative_water = max(0.0, -float(host(minimum.total())))
        created, max_fraction, unfillable = self._fill_column_holes_readings(
            holes
        )
        del holes
        atmospheric_water = float(host(xp.sum(
            self._level_water_mass_total(water_rows)
        )))
        del water_rows
        # The truncated basis still rings slightly below zero after the
        # projection of the filled field and that residual is TOLERATED:
        # every consumer (physics exchange, diagnostics) clamps at its
        # own boundary.  A degree-zero lift that forced the synthesis
        # nonnegative added |global min| to every level each step and
        # drained the surface reservoir in ~100 steps on the first
        # real-data T63 run.
        analyzed = self.transform.project(
            self.transform.contract_waist(waist.close())
        )
        del waist
        fields = list(bundle.atmosphere.fields())
        fields[4] = analyzed
        atmosphere = bundle.atmosphere.with_fields(fields).with_grid_tracers(
            self._park_tracers(tracers)
        )
        # The surface is untouched: the filler closes inside the column.
        return ArwenGlobalState(
            atmosphere, bundle.surface.copy(), bundle.physics_state.copy()
        ), negative_water, largest_negative_tracer, {
            "water_kg_m2": created,
            "max_rescale": max_fraction,
            "unfillable_kg_m2": unfillable,
            "atmospheric_water_kg_m2": atmospheric_water,
        }


    def _global_mean_total_water(self, bundle: ArwenGlobalState) -> float:
        # The global water mean needs only the six water syntheses and dp;
        # the diagnostics() call this replaces built the full grid state
        # (winds, geopotential, temperature) to read this one number.
        #
        # The atmospheric column is a LEVEL SUM, so it is column-local and
        # a band computes its own rows of it; the plane it fills is 5.13 MB
        # at T533 against the 1.15 GiB of levelled water the whole call
        # would have held.  The four other stores are already planes, and
        # the global mean runs once over the finished total.
        xp = self.transform.backend.xp
        sources = self.grid_sources(
            bundle.atmosphere, only=(*WATER_SPECIES, "dp")
        )
        atmospheric = PlaneAccumulator(
            xp, self.transform.grid.shape,
            self.transform.backend.float_dtype,
            name="total_water_atmospheric", exchange=self.pipeline.exchange,
        )
        for rows in self.pipeline.local_slices():
            g, _stack, _names = self.grid_band(sources, rows)
            atmospheric.add_band(
                rows, xp.sum(atmospheric_water_column(g), axis=0)
            )
            del g
        stores = total_water_column(
            bundle, None, xp, atmospheric=atmospheric.plane
        )
        total = self.transform.backend.to_numpy(stores["total"])
        return float(self.transform.grid.global_mean(total))

    def initialize_water_target(self, bundle: ArwenGlobalState) -> float:
        target = self._global_mean_total_water(bundle)
        self._target_total_water_kg_m2 = float(target)
        return float(target)

    def conservation_targets(self) -> tuple[float | None, float | None]:
        """``(mass_pa, total_water_kg_m2)`` the fixers hold, None before
        the first step or initialisation set them."""
        return self._target_mass_pa, self._target_total_water_kg_m2

    def set_conservation_targets(self, mass_pa, total_water_kg_m2) -> None:
        """Swap the fixers' targets.  A caller stepping several states
        through ONE model (the resident ensemble of
        woof.globe.da) restores each state's own conservation
        epoch before its step; the model itself never changes them
        except through the two initialisers above."""
        self._target_mass_pa = None if mass_pa is None else float(mass_pa)
        self._target_total_water_kg_m2 = (
            None if total_water_kg_m2 is None else float(total_water_kg_m2)
        )

    def _fix_total_water(self, bundle: ArwenGlobalState):
        if not self.water_fixer:
            return bundle, 0.0
        if self._target_total_water_kg_m2 is None:
            self.initialize_water_target(bundle)
        current = self._global_mean_total_water(bundle)
        correction = float(self._target_total_water_kg_m2) - current
        if not math.isfinite(correction):
            raise FloatingPointError("global total-water correction is non-finite")
        surface = bundle.surface.copy()
        xp = self.transform.backend.xp
        # The one place the step writes into a member of a COPIED surface.
        # A parked array is never mutated in place (spill.SpilledArray.copy
        # hands back the same slot, because nothing else distinguishes a
        # copy), so the correction is applied to a staged copy and the slot
        # is replaced with the result.  Same expression, same order.
        water = resident(xp, surface.water_kg_m2) + correction
        # A global "any" is exactly associative, so a band folds into it in
        # any order; it goes through the fold so the refusal is the globe's
        # whatever the schedule was.
        if bool(_associative_over(
            xp, "any", water < -1.0e-7,
            name="fix_total_water_surface_deficit",
        )):
            raise FloatingPointError(
                "global total-water repair requires more surface water than available"
            )
        water = xp.maximum(water, 0.0)
        if isinstance(surface.water_kg_m2, SpilledArray):
            surface.water_kg_m2.store(water)
        else:
            surface.water_kg_m2 = water
        del water
        return ArwenGlobalState(
            bundle.atmosphere, surface, bundle.physics_state.copy()
        ), correction

    def _physics_exchange(self, bundle: ArwenGlobalState, dt_s: float):
        exchange, _fixer = self._physics_exchange_with_closure(bundle, dt_s)
        return exchange

    def whole_globe_slices(self):
        """The bands an operator runs when it must produce the GLOBE.

        Two operators need this rather than
        :meth:`BandPipeline.local_slices`, and both leave every card
        holding the whole result out of inputs that are the same on every
        card:

        - the physics half-step, whose OUTPUTS (the grid tracers, the
          surface reservoirs, the native physics namespace) are read whole
          by everything downstream -- the water fixer's global mean, the
          guards, the diagnostics, the checkpoint, the exports, the DA
          door;
        - ``_transport_fluxes``, because the transport takes the globe and
          its four driving fields are synthesised from the REPLICATED
          spectral state.

        THE SUITE IS BANDED (:meth:`apply_physics` runs it a band at a
        time, and a band's half-step holds that band's working set), so on
        one card this is the same loop every other operator runs and the
        band count is what bounds the physics' memory.  On two cards the
        physics is still DUPLICATED rather than split, and the reason is
        its consumers, not the suite: a card that ran only its own bands
        would hold only its own rows of the surface and the namespace,
        and every consumer above reads a whole plane of them.  Splitting
        it costs either a per-half-step gather of the namespace (2.3 GiB
        at T533, 0.8 s over 25 GbE, slower than the duplicated physics)
        or the two-stage rewrite of every one of those consumers, and
        neither is in this tree.  What a second card buys is therefore
        the dynamics, the transport and the right-hand sides, and the
        receipt says so rather than leaving a reader to infer it from a
        disappointing wall clock.
        """
        return self.pipeline.slices()

    def gather_grid_tracers(self, atmosphere):
        """ Bring the ten grid tracers whole before the suite reads them.

        The tracers are the one part of the physics half-step's input that
        is genuinely partitioned: the transport sweeps them band by band
        and the positivity repair floors them band by band, so on a
        two-card run each card holds its own rows and stale values
        elsewhere.  Everything else the exchange is built from -- the
        pressures, the temperatures, the geopotential, the winds -- is
        synthesised from the REPLICATED spectral state, so each card
        computes it whole for itself and no byte crosses the wire for it.

        One gather per physics half-step, ten volumes: 453 MB at T255 L40
        float32 per card per half-step.  It exists because the physics
        half-step is duplicated on every card (:meth:`whole_globe_slices`).
        """
        exchange = self.pipeline.exchange
        if exchange is None or int(getattr(exchange, "world", 1)) <= 1:
            return
        xp = self.transform.backend.xp
        for name, value in atmosphere.grid_tracers().items():
            exchange.fill_rows(xp, value, value.ndim - 2, name=f"tracer_{name}")

    #: What the physics exchange reads of the grid state.  The exchange
    #: never reads grid vorticity or divergence (winds come from the
    #: spectral wind synthesis), so their grid syntheses are not paid;
    #: theta and vapor are the only 3-D spectral syntheses; the ten grid
    #: tracers are the state's own arrays.
    _PHYSICS_GRID_KEYS = (
        "p_half", "p_full", "dp", "temperature", "theta",
        "virtual_temperature", "geopotential", "u", "v",
        *ADVECTED_TRACERS,
    )

    def _physics_sources(self, bundle: ArwenGlobalState, dt_s: float):
        """Everything one physics call reads that is built ONCE and drained
        a band at a time (:class:`_PhysicsSources`).

        Every Legendre contraction the call needs runs here, whole, at the
        operand shapes a resident step gives it (:meth:`grid_sources`); the
        bands drain the waists.  The model-top pressure is read here over
        the WHOLE grid, as the float32 mean of the top half-level plane --
        the expression the radiation driver's column extension took over
        the whole batch -- so that every band's radiation is handed the
        same float: a band's own mean of that plane is not the globe's
        once the sum passes 2^24 of the top pressure's binade, and the
        above-model column is built on it.
        """
        xp = self.transform.backend.xp
        dtype = self.transform.backend.float_dtype
        nlat, nlon = self.transform.grid.shape
        sources = self.grid_sources(bundle.atmosphere, only=self._PHYSICS_GRID_KEYS)
        _spectral, div_mass = self._mass_flux_sources(sources)
        lat = self.transform.backend.asarray(
            self.transform.grid.latitude_deg[:, None], dtype=dtype
        )
        lon = self.transform.backend.asarray(
            self.transform.grid.longitude_deg[None, :], dtype=dtype
        )
        lat = xp.broadcast_to(lat, self.transform.grid.shape)
        lon = xp.broadcast_to(lon, self.transform.grid.shape)
        top = PlaneAccumulator(
            xp, (nlat, nlon), xp.float32, name="physics_model_top")
        for rows in self.whole_globe_slices():
            block = self._pressure_band(sources.pressure, rows)
            top.add_band(rows, xp.asarray(block["p_half"][0], dtype=xp.float32))
            del block
        model_top_pa = float(np.asarray(
            self.transform.backend.to_numpy(top.plane), dtype=np.float32
        ).mean())
        del top
        return _PhysicsSources(
            sources=sources, div_mass=div_mass, lat=lat, lon=lon,
            holes=self._column_hole_accumulators(),
            model_top_pa=model_top_pa, dt_s=float(dt_s),
            time_s=float(bundle.time_s),
        )

    def _physics_exchange_band(self, ps, bundle: ArwenGlobalState, tracers, rows: slice,
                               *, band):
        """One band's :class:`PhysicsExchange`: the rows ``rows`` of the
        globe, built from the call's sources.  ``band`` is what the
        exchange is stamped with: the rows, for the model's loop, whose
        results the model finishes together; None for the stand-alone
        whole-grid exchange a harness drives, which finishes itself.

        ``tracers`` is the band's ten grid tracers, already floored.
        Everything else -- the syntheses, the pressure block, the exner
        power, the virtual temperature, the hydrostatic geopotential, the
        column-hole filler and the continuity closure -- runs here on the
        band's rows, and what the band cannot finish alone (the filler's
        three readings, the level water masses) goes through the call's
        accumulators.  The surface and the namespace reach the band as the
        band's rows of them: a slab of an array the card holds, or a
        staged band of one the pinned host tier holds.
        """
        from .state import SURFACE_ARRAY_NAMES, PhysicsState, SurfaceState, _copy_json

        xp = self.transform.backend.xp
        nlat = self.transform.grid.nlat
        g, stack, names = self.grid_band(ps.sources, rows)
        exner = (g["p_full"] / REFERENCE_PRESSURE_PA) ** KAPPA
        # Physics owns a nonnegative-water contract and its guard refuses
        # negatives it PRODUCED; the spectral representation of vapor
        # rings slightly below zero in grid space at sharp fronts, so the
        # ring is removed at this boundary and closed inside its own
        # column (_fill_column_holes): every column's vapor integral
        # reaches the physics unchanged and no other column pays.  Virtual
        # temperature was formed from the raw vapor inside the synthesis
        # above, before this.
        raw_qv = stack[self._stack_row_index(names, ("qv",))][0]
        filled = self._fill_column_holes_band(raw_qv, g["dp"], ps.holes, rows)
        del raw_qv
        condensate = sum(tracers[name] for name in CONDENSATE_SPECIES)
        band_rows = self._level_water_mass_rows(
            xp.stack([filled, condensate]), g["dp"]
        )
        del condensate
        if ps.water_rows is None:
            ps.water_rows = LatitudeAccumulator(
                xp, (*band_rows.shape[:-1], nlat), band_rows.dtype,
                name="level_water_mass", exchange=self.pipeline.exchange,
            )
        ps.water_rows.add_band(rows, band_rows)
        del band_rows
        # The pressure velocity on the half levels, diagnosed from the
        # same continuity closure rhs() integrates with.  The native
        # suite's cumulus scheme reads it as w = -omega/(rho g); nothing
        # else in the exchange changes, so a suite that never reads it
        # sees bit-identical input.
        _dm, _ps_t, omega = self._mass_flux_band(ps.div_mass, rows)
        del _dm, _ps_t
        surface = SurfaceState(**{
            member: resident(xp, getattr(bundle.surface, member), rows)
            for member in SURFACE_ARRAY_NAMES
        })
        physics_state = PhysicsState(
            schema=bundle.physics_state.schema,
            arrays={
                name: resident(xp, value, rows)
                for name, value in bundle.physics_state.arrays.items()
            },
            metadata=_copy_json(bundle.physics_state.metadata),
        )
        exchange = PhysicsExchange.create(
            time_s=ps.time_s,
            dt_s=ps.dt_s,
            latitude_deg=band_view(xp, ps.lat, rows),
            longitude_deg=band_view(xp, ps.lon, rows),
            p_half=g["p_half"],
            p_full=g["p_full"],
            dp=g["dp"],
            exner=exner,
            temperature=g["temperature"],
            theta=g["theta"],
            virtual_temperature=g["virtual_temperature"],
            geopotential=g["geopotential"],
            u=g["u"],
            v=g["v"],
            qv=filled,
            qc=tracers["qc"], qr=tracers["qr"],
            qi=tracers["qi"], qs=tracers["qs"], qg=tracers["qg"],
            nc=tracers["nc"], nr=tracers["nr"], ni=tracers["ni"],
            ns=tracers["ns"], ng=tracers["ng"],
            surface=surface,
            physics_state=physics_state,
            omega_half_pa_s=omega,
            band=band,
            model_top_pa=ps.model_top_pa,
        )
        del g, stack, exner, filled, omega
        return exchange

    def _physics_fixer_readings(self, ps) -> dict[str, float]:
        """The exchange filler's readings of one call, once over the
        call's accumulators."""
        xp = self.transform.backend.xp
        created, max_fraction, unfillable = self._fill_column_holes_readings(
            ps.holes
        )
        return {
            "water_kg_m2": created,
            "max_rescale": max_fraction,
            "unfillable_kg_m2": unfillable,
            "atmospheric_water_kg_m2": float(
                self.transform.backend.to_numpy(xp.sum(
                    self._level_water_mass_total(ps.water_rows)
                ))
            ),
        }

    def _floor_tracer_bands(self, tracers, minima, maxima, *, what: str):
        """Floor one band's tracers at zero where a band's minimum is below
        it, folding every band's minimum and maximum into the call's.

        The floor is a roundoff instrument (:meth:`_floor_grid_tracers`):
        the fold keeps the globe's minimum and maximum, which is what the
        refusal after the loop reads, and the floor writes only the bands
        that need it, so at one band a nonnegative tracer is the state's
        own array and nothing is copied.  One host read per band.
        """
        xp = self.transform.backend.xp
        host = self.transform.backend.to_numpy
        names = list(tracers)
        band_minima = [xp.min(tracers[name]) for name in names]
        band_maxima = [xp.max(tracers[name]) for name in names]
        for name, low, high in zip(names, band_minima, band_maxima):
            minima[name].add_band(low)
            maxima[name].add_band(high)
        readings = host(xp.stack(band_minima))
        out = dict(tracers)
        for name, minimum in zip(names, readings):
            if float(minimum) < 0.0:
                out[name] = xp.maximum(tracers[name], 0.0)
        return out

    def _refuse_floored_tracers(self, minima, maxima) -> float:
        """The globe's roundoff floor reading: the most negative value any
        tracer carried, and a refusal when a floor took more than roundoff
        of a field's own maximum (:meth:`_floor_grid_tracers`)."""
        xp = self.transform.backend.xp
        host = self.transform.backend.to_numpy
        eps = float(xp.finfo(self.transform.backend.float_dtype).eps)
        names = list(minima)
        readings = host(xp.stack(
            [minima[name].total() for name in names]
            + [maxima[name].total() for name in names]
        ))
        largest = 0.0
        for index, name in enumerate(names):
            minimum = float(readings[index])
            maximum = float(readings[len(names) + index])
            if minimum < 0.0:
                if -minimum > 64.0 * eps * max(maximum, 1.0e-30):
                    raise FloatingPointError(
                        f"grid tracer {name} is negative beyond roundoff "
                        f"(min {minimum:.3g} against max {maximum:.3g}): a "
                        "grid-point tracer is nonnegative by construction, "
                        "so this is a transport or physics defect, not "
                        "representation ringing"
                    )
                largest = max(largest, -minimum)
        return largest

    def _tracer_extrema(self):
        xp = self.transform.backend.xp
        return (
            {name: AssociativeAccumulator(xp, "min", name=f"tracer_min_{name}")
             for name in GRID_TRACERS},
            {name: AssociativeAccumulator(xp, "max", name=f"tracer_max_{name}")
             for name in GRID_TRACERS},
        )

    def _physics_exchange_with_closure(self, bundle: ArwenGlobalState, dt_s: float):
        """The WHOLE-GLOBE exchange the physics suite reads, and the
        filler's readings: the band loop of :meth:`apply_physics` run for
        one band that is the globe.  The verification harnesses and the
        tests drive a suite from this; the step itself never builds it.
        """
        xp = self.transform.backend.xp
        ps = self._physics_sources(bundle, dt_s)
        rows = self.whole_rows
        # The grid tracers reach the physics as they are: nonnegative by
        # construction, no clamp, no rescale, no moment bookkeeping.  The
        # roundoff floor measures them and floors the band that needs it.
        minima, maxima = self._tracer_extrema()
        tracers = self._floor_tracer_bands(
            {name: band_view(xp, value, rows)
             for name, value in bundle.atmosphere.grid_tracers().items()},
            minima, maxima, what="exchange")
        self._refuse_floored_tracers(minima, maxima)
        exchange = self._physics_exchange_band(ps, bundle, tracers, rows, band=None)
        fixer = self._physics_fixer_readings(ps)
        ps.release()
        return exchange, fixer

    def _top_sponge(self, u, v, p_full, p_half, dt_s: float):
        """Graded top-of-model wave absorber: Rayleigh-damp wind
        anomalies toward the zonal mean, cos^2-ramped from the base to
        the lid.

        Moved here from ReferencePhysics (suite v7 -> v8, 2026-08-31):
        the absorber compensates the DYCORE's rigid p_top lid, so it
        belongs to the model, not to whichever physics suite is loaded -
        the full native five-scheme T255 run died at hour 6.55 on the
        140 K research bound (polar-top collapse, 242 m/s top winds)
        because the v7 copy acted only in reference mode.  PLACEMENT:
        once per physics half-step, on the grid winds between the physics
        result and its wind analysis (and on a wind synthesis/analysis
        round trip when no suite is loaded).  That is arithmetically the
        v7 position - the reference suite applied it after turbulence and
        no later reference stage touched winds - so a reference-mode step
        is bit-identical to v7, and every mode receives exp(-half_dt *
        rate) twice per model step, the same total damping v7 delivered.

        Why (measured on the two T533 hour-3 crash states, 2026-08-31):
        the rigid lid reflects a terrain-locked near-truncation gravity-
        wave train (divergence variance peaking at total degree 365-445,
        90-110 km) whose divergence drives adiabatic cooling of
        -0.04..-0.16 K/s; both runs died at the 140 K research gate
        within 3.5 h.  The v6 predecessor damped one level (ring-mean
        p_full < 700 Pa) at 1/21,600 s = 4.6e-5 1/s while the pocket
        amplitude e-folded at 2.36e-4 1/s - 5.1x too slow - and the wave
        occupied four levels (wind-anomaly rms 9.5/9.7/8.2 m/s at levels
        0-2, 8-9 K cold pockets at levels 2-3), so it removed 26% of one
        level's wind anomaly in 3 h and never touched the structure that
        regenerates it.  The graded form covers every ring with ring-mean
        p_full below sponge_base_pa (5000 Pa = every ring of levels 0-2
        plus 147 of the 801 level-3 rings on the T533 20-level stack;
        ring-mean p_full at the crash-pocket latitude 257/1133/2693/4766
        Pa) at rate
        cos^2((pi/2)(p - p_lid)/(base - p_lid)) / tau_lid: 1.11e-3 1/s at
        the top ring (4.7x the measured growth), smoothly zero at the
        base.

        What is damped: only the deviation of u and v from each ring's
        instantaneous zonal mean - the jet is never touched - and the
        factor is uniform around each (level, latitude) ring, so the
        damped anomalies sum to zero over the ring.  Because divergence
        is linear in the wind and the factor is ring-uniform, the eddy
        divergence - the measured driver (pocket divergence -9.8e-4 1/s
        = 5.7x window rms) - is damped at the same local rate; this is
        the diagnosis's divergence damping, delivered through the
        momentum anomalies rather than a separate spectral operator.

        Measured on the v6 T533 hour-3 crash state over a 30-minute
        absorber-only window (60 steps of dt=30 s, shipping defaults):
        wind-anomaly rms removed 86.4/82.7/54.3/0.8% at levels 0-3;
        pocket finite-difference divergence -4.1e-4 -> -6.3e-5 1/s
        (surviving fraction 0.155 vs 0.136 predicted from the ring's
        rate), against a runaway whose amplitude e-folds in 4230 s.
        Conservation, same window: every ring's zonal-mean wind is
        conserved by construction (max ring-mean drift 1.5e-5 m/s on
        the run's own 801x1602 grid = roundoff at fp32 wind
        amplitudes); the dp-weighted ring momentum
        is a stated limit, not conserved where dp varies along a ring
        (4.6e-2 relative at the top level on the crash state); eddy
        kinetic energy is removed with no heat booking (-43.9 J/kg at
        the top level, 98% of its eddy KE) - that removal is the
        absorber's purpose, standing in for the absorption a rigid lid
        cannot provide.  Momentum only; no temperature, no water.
        """
        if self.sponge_base_pa <= 0.0:
            return u, v
        xp = self.transform.backend.xp
        # The ring-mean pressure gates each (level, latitude) ring as a
        # unit: a per-point gate would damp part of a ring toward the
        # full-ring mean and break the zero-sum above wherever surface
        # pressure varies along the ring.
        ring_p = xp.mean(p_full, axis=-1, keepdims=True)
        ring_lid = xp.mean(p_half[0], axis=-1, keepdims=True)
        in_sponge = ring_p < self.sponge_base_pa
        # cos^2 ramp: full rate at the lid, smoothly zero at the base.
        # The clip guards the degenerate base <= lid configuration, where
        # in_sponge is empty anyway (p_full always exceeds the lid).
        ramp = xp.clip(
            (ring_p - ring_lid)
            / xp.maximum(self.sponge_base_pa - ring_lid, 1.0e-3),
            0.0,
            1.0,
        )
        rate = (
            xp.cos(0.5 * math.pi * ramp) ** 2
            / self.sponge_lid_relaxation_time_s
        )
        # Exact integral of du/dt = -rate * (u - ubar) over dt, stable
        # for any dt * rate; first order it removes the dt * rate
        # fraction of the anomaly per step.
        decay = xp.exp(-float(dt_s) * rate)
        u_mean = xp.mean(u, axis=-1, keepdims=True)
        v_mean = xp.mean(v, axis=-1, keepdims=True)
        u = xp.where(in_sponge, u_mean + (u - u_mean) * decay, u)
        v = xp.where(in_sponge, v_mean + (v - v_mean) * decay, v)
        return u, v

    def _sponge_only_physics_pass(self, bundle: ArwenGlobalState, dt_s: float):
        """The lid absorber with no physics suite loaded.

        With physics attached the winds already round-trip through grid
        space each half-step, so the absorber rides that trip for free;
        with no suite the trip exists only for the absorber, and it is
        paid only when a ring actually sits inside the sponge - a deep-lid
        configuration (every ring-mean p_full >= base) returns the bundle
        untouched, so trajectories that the absorber cannot affect do not
        pick up wind analysis/synthesis roundoff.
        """
        xp = self.transform.backend.xp
        if self.sponge_base_pa <= 0.0:
            return bundle, {"physics_mode": "none"}
        # Whether ANY ring sits inside the sponge is an "any" over the
        # globe, exact in any order, and it is asked BEFORE a wind is
        # synthesised so a deep-lid configuration pays nothing.
        pressure = self._pressure_sources(bundle.atmosphere)
        inside = AssociativeAccumulator(xp, "any", name="sponge_any_ring", exchange=self.pipeline.exchange)
        for rows in self.whole_globe_slices():
            block = self._pressure_band(pressure, rows)
            inside.add_band(
                xp.mean(block["p_full"], axis=-1, keepdims=True)
                < self.sponge_base_pa
            )
            del block
        if not bool(self.transform.backend.to_numpy(inside.total())):
            return bundle, {"physics_mode": "none"}
        sources = self.grid_sources(
            bundle.atmosphere, only=("p_half", "p_full", "u", "v")
        )
        waist = self.transform.open_waist(
            (2, self.nlev), bands=self.pipeline.bands
        )
        for rows in self.whole_globe_slices():
            g, _stack, _names = self.grid_band(sources, rows)
            u, v = self._top_sponge(
                g["u"], g["v"], g["p_full"], g["p_half"], float(dt_s)
            )
            pair = xp.empty(
                (2, *u.shape), dtype=self.transform.backend.float_dtype
            )
            pair[0] = u
            pair[1] = v
            del u, v, g
            waist.fill_band(rows.start, rows.stop, pair)
            del pair
        zeta, divergence = self.vector.vordiv_from_wind_waist(waist.close())
        del waist, sources
        fields = list(bundle.atmosphere.fields())
        fields[0] = zeta
        fields[1] = divergence
        atmosphere = bundle.atmosphere.with_fields(fields)
        return ArwenGlobalState(
            atmosphere, bundle.surface.copy(), bundle.physics_state.copy()
        ), {"physics_mode": "none"}

    #: The four physics-record rows the step SUMS over a half's suite calls;
    #: every other row is the last call's (the diagnostics of the state the
    #: half hands on).
    _PHYSICS_SUMMED_ROWS = (
        "exchange_fixer_water_kg_m2", "exchange_fixer_unfillable_kg_m2",
    )
    _PHYSICS_MAX_ROWS = (
        "exchange_fixer_max_rescale", "exchange_atmospheric_water_kg_m2",
        "physics_result_floored_negative",
    )

    def _apply_physics_half(self, bundle: ArwenGlobalState, dt_s: float):
        """One Strang half of the physics, paid in ``semilag.physics_substeps``
        suite calls on the semi-Lagrangian path and in exactly one call
        everywhere else.

        The Eulerian paths never read the option: the loop below runs once
        with the same arguments ``apply_physics`` always received, so the
        arithmetic of every IMEX and split-era step is the arithmetic it was
        (BIT-IMEX).  On the semi-Lagrangian path with ``n > 1`` the half is
        ``n`` calls of ``dt_s / n`` on the same land and radiation buckets,
        which is the instrument that holds the dynamical step at 300 s and
        gives the suite the call length the Eulerian core gives it.
        """
        count = 1
        if self.semi_lagrangian:
            count = int(self.semilag.physics_substeps)
        if count <= 1:
            return self.apply_physics(bundle, dt_s)
        slab = float(dt_s) / count
        summed = {name: 0.0 for name in self._PHYSICS_SUMMED_ROWS}
        largest = {name: 0.0 for name in self._PHYSICS_MAX_ROWS}
        record: dict = {}
        for _ in range(count):
            bundle, record = self.apply_physics(bundle, slab)
            for name in summed:
                summed[name] += float(record.get(name, 0.0))
            for name in largest:
                largest[name] = max(largest[name], float(record.get(name, 0.0)))
        record = dict(record)
        record.update(summed)
        record.update(largest)
        record["physics_substeps"] = int(count)
        return bundle, record

    def apply_physics(self, bundle: ArwenGlobalState, dt_s: float):
        """One physics call, a latitude band at a time.

        THE BAND IS THE OUTER LOOP.  For every band of the schedule the
        exchange is built for that band's rows, the suite runs on it, and
        its result goes where the globe's result goes: the sponged winds
        and the theta/vapor pair into full-latitude Fourier waists (two
        whole contractions after the loop, at the operand shapes the
        resident return gives them), the grid tracers, the surface and the
        namespace into their whole-grid destinations (:class:`_GridAssembler`:
        the array itself at one band, the pinned host tier's slot band by
        band when it holds the slice, a device buffer otherwise).  So a
        band's half-step allocates only that band's working set -- the
        exchange, the batch, the schemes' workspaces and the result, all at
        the band's width -- and the persistent state the tier holds reaches
        the card a band at a time.  The suite's per-grid state (the
        radiation's solar geometry, the frozen-column masks, the cumulus
        grid spacing) is keyed on the band inside the suite.

        Every reading the call reports comes out the same at every band
        count (physics.banding): the maxima and counts fold exactly, the
        grid means are reduced once over planes the bands fill, the
        radiation size-bounding record is assembled from per-column sums,
        and anything else is checked equal across the bands.  At one band
        the loop runs once over the globe and is the resident call.
        """
        if float(dt_s) == 0.0:
            return bundle, {"physics_mode": "none"}
        if self.physics is None:
            return self._sponge_only_physics_pass(bundle, dt_s)
        from .state import SURFACE_ARRAY_NAMES, PhysicsState, SurfaceState, _copy_json

        prof = profiler_of(self)
        xp = self.transform.backend.xp
        dtype = self.transform.backend.float_dtype
        nlat, nlon = self.transform.grid.shape
        dt = float(dt_s)
        with prof.section("exchange"):
            ps = self._physics_sources(bundle, dt)
        # The suite call is the step's device peak: the memo's syntheses
        # are released first so the peak is the one the memo-less step
        # had.  ``ps`` keeps its own references to the call's waists.
        self.release_syntheses()
        state_tracers = bundle.atmosphere.grid_tracers()
        in_min, in_max = self._tracer_extrema()
        out_min, out_max = self._tracer_extrema()
        out_tracers = _GridAssembler(
            self, "tracer", state_tracers, parked=self._spills("tracers"))
        out_surface = _GridAssembler(
            self, "surface",
            {member: getattr(bundle.surface, member) for member in SURFACE_ARRAY_NAMES},
            parked=self._spills("surface"))
        out_physics = _GridAssembler(
            self, "physics", bundle.physics_state.arrays,
            parked=self._spills("physics"))
        planes: dict[str, PlaneAccumulator] = {}
        band_diagnostics: list[dict] = []
        band_metadata: list[dict] = []
        columns: list[int] = []
        adapter_receipt: dict = {}
        wind_waist = self.transform.open_waist(
            (2, self.nlev), bands=self.pipeline.bands
        )
        limit = max(1, int(self.spectral_chunk))
        scalar_waists = [
            (start, self.transform.open_waist(
                (min(limit, 2 - start), self.nlev), bands=self.pipeline.bands
            ))
            for start in range(0, 2, limit)
        ]
        for rows in self.whole_globe_slices():
            with prof.section("exchange"):
                # The grid tracers reach the physics as they are:
                # nonnegative by construction, no clamp, no rescale, no
                # moment bookkeeping.  The roundoff floor measures the
                # band and floors it only where it has to, so at one band a
                # nonnegative tracer is the state's own array.
                tracers = self._floor_tracer_bands(
                    {name: band_view(xp, value, rows)
                     for name, value in state_tracers.items()},
                    in_min, in_max, what="exchange")
                exchange = self._physics_exchange_band(
                    ps, bundle, tracers, rows,
                    band=(int(rows.start), int(rows.stop)))
                del tracers
                # The derived volumes of this band (temperature, virtual
                # temperature, geopotential) die with the exchange rather
                # than waiting for the next band to evict them.
                self._band_values.clear()
                self._band_tag = None
            with prof.section("suite"):
                result = self.physics.step(exchange)
            with prof.section("return"):
                # The lid absorber acts on the physics result's grid winds
                # before their analysis: arithmetically the suite-v7
                # position (after turbulence, nothing later in the reference
                # suite touched winds), so reference mode is bit-identical to
                # v7 and every suite -- the arwen-native five-scheme stack
                # included -- is protected from the lid reflection the
                # absorber exists to remove.  The absorber is per (level,
                # latitude) ring: every one of its reductions is a zonal
                # mean inside one row.
                sponged_u, sponged_v = self._top_sponge(
                    result.u, result.v, exchange.p_full, exchange.p_half, dt,
                )
                pair = xp.empty((2, *sponged_u.shape), dtype=dtype)
                pair[0] = sponged_u
                pair[1] = sponged_v
                del sponged_u, sponged_v
                wind_waist.fill_band(rows.start, rows.stop, pair)
                del pair
                stack = xp.empty(
                    (2, self.nlev, rows.stop - rows.start, nlon), dtype=dtype,
                )
                stack[0] = xp.asarray(result.theta, dtype=dtype)
                stack[1] = xp.asarray(result.qv, dtype=dtype)
                for start, waist in scalar_waists:
                    waist.fill_band(
                        rows.start, rows.stop, stack[start : start + limit]
                    )
                del stack
                # The grid tracers come back as the physics result's own
                # grid arrays: no analysis, no ringing, no clamp beyond the
                # roundoff floor.
                floored = self._floor_tracer_bands(
                    {name: xp.asarray(getattr(result, name), dtype=dtype)
                     for name in GRID_TRACERS},
                    out_min, out_max, what="result")
                for name, value in floored.items():
                    out_tracers.put(name, rows, value)
                del floored
                for member in SURFACE_ARRAY_NAMES:
                    out_surface.put(member, rows, getattr(result.surface, member))
                for name, value in result.physics_state.arrays.items():
                    out_physics.put(name, rows, value)
                for name, value in result.planes.items():
                    accumulator = planes.get(name)
                    if accumulator is None:
                        accumulator = planes[name] = PlaneAccumulator(
                            xp, (*value.shape[:-2], nlat, int(value.shape[-1])),
                            value.dtype, name=f"physics_plane_{name}",
                        )
                    accumulator.add_band(rows, value)
                band_diagnostics.append(dict(result.diagnostics))
                band_metadata.append(dict(result.physics_state.metadata))
                columns.append((rows.stop - rows.start) * nlon)
                adapter_receipt = result.adapter_receipt
                del result, exchange
        with prof.section("return"):
            largest_negative = self._refuse_floored_tracers(out_min, out_max)
            self._refuse_floored_tracers(in_min, in_max)
            del in_min, in_max, out_min, out_max
            exchange_fixer = self._physics_fixer_readings(ps)
            ps.release()
            del ps
            zeta, divergence = self.vector.vordiv_from_wind_waist(
                wind_waist.close()
            )
            del wind_waist
            analyzed = None
            for start, waist in scalar_waists:
                piece = self.transform.contract_waist(waist.close())
                if analyzed is None and len(scalar_waists) == 1:
                    analyzed = piece
                    break
                if analyzed is None:
                    analyzed = xp.empty((2, *piece.shape[1:]), dtype=piece.dtype)
                analyzed[start : start + piece.shape[0]] = piece
                del piece
            del scalar_waists
            surface = SurfaceState(**out_surface.finish())
            physics_arrays = out_physics.finish()
            tracers = out_tracers.finish()
            atmosphere = bundle.atmosphere.with_fields(
                (
                    zeta,
                    divergence,
                    analyzed[0],
                    bundle.atmosphere.log_surface_pressure,
                    analyzed[1],
                )
            ).with_grid_tracers(tracers)
            del zeta, divergence, analyzed, tracers
        with prof.section("finish"):
            # The call's readings, once, over the assembled globe.  A suite
            # that declares no merge of its own (a harness's test physics)
            # gets the default: every reading the same on every band.
            finish = getattr(self.physics, "finish", None)
            if not callable(finish):
                from .physics.banding import default_finish

                finish = default_finish
            diagnostics, metadata = finish(
                band_diagnostics, band_metadata,
                {name: accumulator.plane for name, accumulator in planes.items()},
                surface,
                PhysicsState(
                    schema=bundle.physics_state.schema, arrays=physics_arrays,
                    metadata=_copy_json(bundle.physics_state.metadata),
                ),
                metadata_in=bundle.physics_state.metadata,
                columns=columns, dt_s=dt,
            )
            del planes
        physics_state = PhysicsState(
            schema=bundle.physics_state.schema, arrays=physics_arrays,
            metadata=metadata,
        )
        physics_state.validate()
        # The exchange filler closed the vapor clamp inside each column
        # before physics ran, so the surface the physics returns is the
        # surface the model carries: no clamp touches the reservoir.
        return ArwenGlobalState(
            atmosphere, surface, physics_state
        ), {
            "physics_mode": adapter_receipt.get("mode", "unknown"),
            "physics": diagnostics,
            "physics_adapter": adapter_receipt,
            "exchange_fixer_water_kg_m2": float(exchange_fixer["water_kg_m2"]),
            "exchange_fixer_max_rescale": float(exchange_fixer["max_rescale"]),
            "exchange_fixer_unfillable_kg_m2": float(
                exchange_fixer["unfillable_kg_m2"]
            ),
            "exchange_atmospheric_water_kg_m2": float(
                exchange_fixer["atmospheric_water_kg_m2"]
            ),
            "physics_result_floored_negative": float(largest_negative),
        }

    def enforce(self, bundle: ArwenGlobalState) -> None:
        xp = self.transform.backend.xp
        host = self.transform.backend.to_numpy
        # Every reduction of the guards runs where the arrays live and the
        # scalars cross to the host in ONE read (thirty-one reads and
        # synchronisations per step before, profile 2026-09-04); the
        # checks then run in the same order with the same sentences.
        spectral_finite = [
            xp.all(xp.isfinite(x.real)) & xp.all(xp.isfinite(x.imag))
            for x in bundle.atmosphere.fields()
        ]
        # The guards below read temperature, surface pressure, vapor and
        # dp - not winds, geopotential, or the grid vorticity/divergence,
        # so those syntheses are not paid here.  Every one of them folds
        # over the bands: four maxima and minima, ten finiteness tests and
        # ten tracer minima are exactly associative, and the two vapor
        # mass readings go into the resident (level, latitude) buffer
        # below.
        sources = self.grid_sources(
            bundle.atmosphere, only=("qv", "temperature", "ps", "dp"),
        )
        folds = {
            name: AssociativeAccumulator(xp, op, name=f"enforce_{name}", exchange=self.pipeline.exchange)
            for name, op in (
                ("t_min", "min"), ("t_max", "max"),
                ("ps_min", "min"), ("ps_max", "max"),
            )
        }
        tracer_folds = []
        for name in GRID_TRACERS:
            tracer_folds.append((
                name,
                AssociativeAccumulator(
                    xp, "all", name=f"enforce_{name}_finite", exchange=self.pipeline.exchange),
                AssociativeAccumulator(xp, "min", name=f"enforce_{name}_min", exchange=self.pipeline.exchange),
            ))
        # The vapor negative- and positive-mass readings were the ONE flat
        # three-dimensional sum left in the step, and a flat sum has no
        # band-by-band form that keeps its last bits.  They are two-stage
        # here: the band-local stage reduces LONGITUDE ONLY, into a
        # resident (level, latitude) buffer a band fills for exactly its
        # own rows, and the whole stage sums that fixed-shape buffer once.
        #
        # Why the level axis stays in the buffer rather than being folded
        # per band into an nlat vector: MEASURED 2026-09-06, a reduction's
        # own algorithm moves with the SHAPE it is handed -- numpy sums a
        # (nlev, 1) column pairwise and a (nlev, n>1) block sequentially --
        # so a one-row band computed a different level sum from the same
        # row inside a wider band, and gate RED-1 refused at B = 64 and
        # B = nlat.  A buffer whose shape does not depend on the band count
        # keeps the varying stage down to the one axis that is row-local.
        #
        # The readings move by a few ulp against the flat sum they replace;
        # they feed the representation-ringing refusal below and nothing
        # that is checkpointed, and the checkpoint inventory does not move.
        row_shape = (self.nlev, self.transform.grid.nlat)
        negative_rows = None
        positive_rows = None
        for rows in self.pipeline.local_slices():
            g, _stack, _names = self.grid_band(sources, rows)
            folds["t_min"].add_band(g["temperature"])
            folds["t_max"].add_band(g["temperature"])
            folds["ps_min"].add_band(g["ps"])
            folds["ps_max"].add_band(g["ps"])
            for name, finite, minimum in tracer_folds:
                value = band_view(xp, getattr(bundle.atmosphere, name), rows)
                finite.add_band(xp.isfinite(value))
                minimum.add_band(value)
                del value
            weighted = g["qv"] * g["dp"]
            if negative_rows is None:
                negative_rows = LatitudeAccumulator(
                    xp, row_shape, weighted.dtype, name="enforce_qv_negative", exchange=self.pipeline.exchange,
                )
                positive_rows = LatitudeAccumulator(
                    xp, row_shape, weighted.dtype, name="enforce_qv_positive", exchange=self.pipeline.exchange,
                )
            negative_rows.add_band(
                rows, xp.sum(xp.minimum(weighted, 0.0), axis=-1)
            )
            positive_rows.add_band(
                rows, xp.sum(xp.maximum(weighted, 0.0), axis=-1)
            )
            del weighted, g
        del sources
        tracer_readings = []
        for _name, finite, minimum in tracer_folds:
            tracer_readings.append(finite.total())
            tracer_readings.append(minimum.total())
        readings = host(xp.concatenate([
            xp.stack(spectral_finite).astype(xp.float64),
            xp.stack([
                folds["t_min"].total(), folds["t_max"].total(),
                folds["ps_min"].total(), folds["ps_max"].total(),
                -negative_rows.total(axis=None),
                positive_rows.total(axis=None),
            ]).astype(xp.float64),
            xp.stack(tracer_readings).astype(xp.float64),
        ]))
        del negative_rows, positive_rows, folds, tracer_folds, tracer_readings
        n_spectral = len(spectral_finite)
        if not all(readings[:n_spectral] != 0.0):
            raise FloatingPointError("WOOF global spectral state contains non-finite values")
        t_min, t_max, ps_min, ps_max, negative, positive = (
            float(v) for v in readings[n_spectral : n_spectral + 6]
        )
        tracer_values = readings[n_spectral + 6 :]
        if t_min < 140.0 or t_max > 380.0:
            raise FloatingPointError(
                f"temperature outside research bounds: {t_min:g}..{t_max:g} K"
            )
        if ps_min < 30_000.0 or ps_max > 120_000.0:
            raise FloatingPointError(
                f"surface pressure outside research bounds: {ps_min:g}..{ps_max:g} Pa"
            )
        # Vapor is the one water field in the spectral basis: it
        # legitimately rings below zero in grid space and every consumer
        # clamps at its own boundary, so the guard separates ringing from
        # a real transport/physics defect by the NEGATIVE-MASS FRACTION,
        # which is scale-free: projecting the kinked sparse morphology
        # saturation produces (max(0, q-qsat)) measures at most 0.28
        # negative/positive mass (T31..T85, worst of 24 trials), while a
        # sign defect drives the ratio toward parity.  Point-amplitude
        # bounds cannot make this cut (the same measurement rings 13-18%
        # of the field maximum), and a 15%-of-maximum guard refused a
        # physics-legal state on the first real-data T63 run.
        if negative > 0.6 * positive and negative > 1.0e-10 * max(positive, 1.0):
            raise FloatingPointError(
                f"qv negative mass is {negative:.3g} against positive "
                f"{positive:.3g}, beyond the 0.6 representation-ringing "
                "bound (measured legitimate worst 0.28)"
            )
        # The grid tracers have no representation ringing to tolerate:
        # the flux-form transport is positive-definite and every physics
        # suite floors its result, so a negative value is a defect and is
        # refused exactly (the 0.6 bound above once covered them; it is
        # retired for them with the spectral representation, 2026-09-02).
        for index, name in enumerate(GRID_TRACERS):
            if not bool(tracer_values[2 * index] != 0.0):
                raise FloatingPointError(
                    f"grid tracer {name} contains non-finite values"
                )
            minimum = float(tracer_values[2 * index + 1])
            if minimum < 0.0:
                raise FloatingPointError(
                    f"grid tracer {name} is negative (min {minimum:.3g}): a "
                    "grid-point tracer is nonnegative by construction, so a "
                    "negative value is a transport or physics defect"
                )
        bundle.physics_state.validate()

    def _transport_fluxes(self, state: MoistHybridState):
        """``(dp, dp u, dp v, omega_half)`` of a spectral state for the
        tracer transport: the layer thickness, the cell-centred mass
        fluxes and the continuity's pressure velocity, every one from the
        memo when the state's arrays were already read this step (the
        CFL gate's wind, the first stage's mass-flux divergence)."""
        xp = self.transform.backend.xp
        sources = self.grid_sources(state, only=("u", "v", "dp"))
        _spectral, div_mass = self._mass_flux_sources(sources)
        if self.pipeline.resident:
            g, _stack, _names = self.grid_band(sources, self.whole_rows)
            dp = g["dp"]
            flux_u = dp * g["u"]
            flux_v = dp * g["v"]
            _dm, _ps_t, omega_half = self._mass_flux_band(
                div_mass, self.whole_rows
            )
            return dp, flux_u, flux_v, omega_half
        # The transport takes the globe: its meridional sweep spans the
        # band edges and its own pipeline cuts them again.  So the four
        # fields are ASSEMBLED here band by band into the arrays it reads,
        # and only one band's grid state is live while they fill.
        shape = (self.nlev, self.transform.grid.nlat, self.transform.grid.nlon)
        dtype = self.transform.backend.float_dtype
        dp = xp.empty(shape, dtype=dtype)
        flux_u = xp.empty(shape, dtype=dtype)
        flux_v = xp.empty(shape, dtype=dtype)
        omega_half = xp.empty((self.nlev + 1, *shape[1:]), dtype=dtype)
        # Every band, not this card's bands: the four fields are the
        # GLOBE's, the transport reads them whole, and they are
        # synthesised from the replicated spectral state, so a card that
        # builds all of them duplicates arithmetic and moves no bytes
        # where a card that built a partition would have to gather it.
        for rows in self.whole_globe_slices():
            g, _stack, _names = self.grid_band(sources, rows)
            dp[..., rows, :] = g["dp"]
            xp.multiply(g["dp"], g["u"], out=flux_u[..., rows, :])
            xp.multiply(g["dp"], g["v"], out=flux_v[..., rows, :])
            _dm, _ps_t, band_omega = self._mass_flux_band(div_mass, rows)
            omega_half[..., rows, :] = band_omega
            del g, _dm, band_omega
        return dp, flux_u, flux_v, omega_half

    def _transport_grid_tracers(
        self, start: MoistHybridState, end: MoistHybridState, dt_s: float,
        start_fluxes=None,
    ):
        """Advance the grid tracers over the step with the flux-form
        transport, driven by the mass fluxes and pressure velocity of the
        dynamics averaged between the step's start state and its
        advanced state (trapezoidal in time).

        ``start`` holds the tracers to transport (the state the dynamics
        started from); ``end`` is the advanced spectral state the tracers
        ride into, whose own ``dp`` is what the transported mixing ratios
        are measured against.  ``start_fluxes`` is ``_transport_fluxes``
        of ``start`` when the caller took it before the dynamics (the
        step does: the first stage's right-hand side reads the same
        syntheses, and after the dynamics the memo holds the stage
        states' instead); None diagnoses it here.  Returns ``(end with
        the transported tracers, metrics)``; the metrics carry the
        sub-cycle counts, the largest Courant number per direction, the
        roundoff floor, and the largest relative gap between the
        transport's pseudo-density and the continuity's ``dp`` (the two
        continuity discretizations' truncation difference, which the
        mixing ratios then ride in).
        """
        xp = self.transform.backend.xp
        host = self.transform.backend.to_numpy
        prof = profiler_of(self)
        with prof.section("fluxes"):
            if start_fluxes is None:
                start_fluxes = self._transport_fluxes(start)
            dp_start, flux_u, flux_v, omega_half = start_fluxes
            del start_fluxes
            dp_end, flux_u_end, flux_v_end, omega_end = self._transport_fluxes(end)
            flux_u = 0.5 * (flux_u + flux_u_end)
            flux_v = 0.5 * (flux_v + flux_v_end)
            omega_half = 0.5 * (omega_half + omega_end)
            del flux_u_end, flux_v_end, omega_end
        with prof.section("advance"):
            tracers, metrics = self.transport.advance(
                start.grid_tracers(), dp_start, flux_u, flux_v, omega_half,
                float(dt_s), step=int(start.step),
            )
        del flux_u, flux_v, omega_half
        pseudo = metrics.pop("pseudo_density")
        gap = xp.abs(pseudo - dp_end) / dp_end
        # The area-weighted global mean of the same gap: the largest gap
        # sits in the sub-cycled polar rows, and the mean says what the
        # atmosphere as a whole rides in.  Both readings cross in one
        # host read.
        cell = self.transform.backend.asarray(
            self.transform.grid.quadrature_weights,
            dtype=self.transform.backend.float_dtype,
        )[None, :, None] / (2.0 * pseudo.shape[-1])
        # The mean is the step's SECOND flat three-dimensional sum, and it
        # was missed by the reduction inventory: a flat sum over
        # (level, latitude, longitude) has no band-by-band form that keeps
        # its last bits, so it takes the same two-stage shape as the vapor
        # mass readings.  The band-local stage reduces LONGITUDE ONLY into
        # a resident (level, latitude) buffer (128 KB at T533) and the
        # whole stage sums that fixed-shape buffer once, because a
        # reduction's own algorithm moves with the shape it is handed.
        # The maximum beside it is exactly associative and folds in any
        # order.  Both readings are transport metrics: they reach the
        # receipt and the ledger, and neither is checkpointed.
        gap_rows = LatitudeAccumulator(
            xp, (gap.shape[0], gap.shape[-2]), gap.dtype,
            name="pseudo_density_gap",
        )
        gap_rows.add_band(slice(0, gap.shape[-2]), xp.sum(gap * cell, axis=-1))
        readings = host(xp.stack([
            _associative_over(xp, "max", gap, name="pseudo_density_gap_max"),
            gap_rows.total(axis=None) / pseudo.shape[0],
        ]))
        metrics["pseudo_density_mismatch_relative"] = float(readings[0])
        metrics["pseudo_density_mismatch_mean_relative"] = float(readings[1])
        del pseudo, gap, dp_end, dp_start
        return end.with_grid_tracers(self._park_tracers(tracers)), metrics

    def step(self, bundle: ArwenGlobalState, dt_s: float):
        if self._target_mass_pa is None:
            self.initialize_mass_target(bundle.atmosphere)
        if self.water_fixer and self._target_total_water_kg_m2 is None:
            # Both conservation targets anchor at the caller's pre-step state.
            # _fix_total_water runs after the step has advanced, so its lazy
            # fallback would bake the first step's drift into the target.
            self.initialize_water_target(bundle)
        half = 0.5 * float(dt_s)
        # The energy ledger's read-only marks around every operator
        # (insitu.energy); a no-op unless the observer samples this step.
        mark = None
        if self.observer is not None:
            mark = getattr(self.observer, "mark_energy", None)
            if mark is not None and not self.observer.begin_energy_step(bundle.atmosphere):
                mark = None
        prof = profiler_of(self)
        merged = self.physics_split == "merged"
        if merged:
            # No physics before the dynamics (physics_split): the state
            # enters the dynamics as it left the previous step's repair,
            # and the ledger's two leading marks read zero.
            first, first_physics = bundle, {"physics_mode": "none"}
            first_negative_water = first_negative_tracer = 0.0
            first_fixer = {
                "water_kg_m2": 0.0, "max_rescale": 0.0,
                "unfillable_kg_m2": 0.0, "atmospheric_water_kg_m2": 0.0,
            }
            if mark is not None:
                mark("physics_first", first.atmosphere)
                mark("positivity_first", first.atmosphere)
        else:
            with prof.section("physics_first"):
                first, first_physics = self._apply_physics_half(bundle, half)
            if mark is not None:
                mark("physics_first", first.atmosphere)
            with prof.section("positivity_first"):
                first, first_negative_water, first_negative_tracer, first_fixer = (
                    self._repair_positivity(first)
                )
            if mark is not None:
                mark("positivity_first", first.atmosphere)
        semi_lagrangian = self.semi_lagrangian
        if semi_lagrangian and self.semilag.physics_coupling != "advected":
            # The state the first half started from, so the semi-Lagrangian
            # core can separate that half's increment from the state and
            # apply a fraction of it at the arrival point instead of
            # letting the gather read all of it at the departure point
            # (semilag.step, semilag.options.PHYSICS_COUPLINGS).  A
            # reference to a state this step holds to its end: no copy.
            self.hold_pre_physics(bundle.atmosphere)
        with prof.section("cfl"):
            cfl = self.cfl(first.atmosphere, dt_s)
        # G1.  The advective CFL is a REFUSAL on every Eulerian path and a
        # MEASUREMENT on the semi-Lagrangian one.  What it guards is the
        # explicit Eulerian advection, which the semi-Lagrangian core does
        # not have: at T255 with a 100 m/s jet this number refuses any dt
        # above 187 s, and the step this integrator exists to take is 300.
        # The refusal that replaces it is the Lipschitz gate inside
        # semilag_step, which names the breakage this one cannot see (a
        # folded trajectory map, which does not blow the model up, it
        # mislocates it silently).  The value is still measured and still
        # written to the receipt on both paths.
        if not semi_lagrangian and cfl > self.maximum_cfl:
            # The refusal names the step this flow would have admitted, at
            # the gate and at the shipped rule, so a day stronger than any
            # the shipped step was set against is explained and not just
            # stopped; the receipt's ``cfl`` block carries the same numbers.
            raise ValueError(spectral_cfl_refusal(
                cfl, dt_s, self.maximum_cfl, self.transform.truncation,
                self.transform.grid.radius_m,
            ))
        self._refuse_beyond_implicit_neutral_limit(dt_s)
        # G3.  The grid tracers ride the semi-Lagrangian stencil unless a
        # config asks for the flux form.  The flux-form sweep sub-cycles
        # to a per-face Courant number of 0.25, and on the outermost T255
        # ring (326 m) a 10 m/s zonal wind at dt = 300 s needs 37
        # sub-steps; at T533 (75 m) it needs 160.  transport.py is not
        # edited: its path stays selectable through [semilag]
        # tracer_scheme = "flux_form" for the arm that compares the two.
        eulerian_tracers = (
            not semi_lagrangian or self.semilag.tracer_scheme == "flux_form"
        )
        # The transport's start fluxes are the syntheses the CFL gate and
        # the first stage read (the memo serves them); taken here, before
        # the stage states replace them in the memo, and held through the
        # dynamics (four grid volumes, 0.19 GiB at T255 float32).
        start_fluxes = None
        if eulerian_tracers:
            with prof.section("transport"):
                with prof.section("fluxes"):
                    start_fluxes = self._transport_fluxes(first.atmosphere)
        with prof.section("dynamics"):
            advanced, semi = self.integrate_dynamics(first.atmosphere, dt_s, mark=mark)
        with prof.section("diffusion"):
            advanced = self._apply_diffusion(advanced, dt_s)
        if mark is not None:
            mark("diffusion", advanced)
        with prof.section("mass_fixer"):
            advanced, mass_offset = self._fix_mass(advanced)
        if mark is not None:
            mark("mass_fixer", advanced)
        # The grid tracers rode through the spectral maps by reference;
        # they are advanced here, once, by the flux-form transport with
        # the step's own mass fluxes.
        if eulerian_tracers:
            with prof.section("transport"):
                advanced, transport = self._transport_grid_tracers(
                    first.atmosphere, advanced, dt_s, start_fluxes=start_fluxes,
                )
            if mark is not None:
                mark("tracer_transport", advanced)
        else:
            # Already advanced, on the step's own stencil, inside the
            # dynamics branch; its metrics ride in that branch's return.
            transport = semi.pop("tracer_transport")
        del start_fluxes
        advanced.step = bundle.step + 1
        advanced.time_s = bundle.time_s + float(dt_s)
        # The first half's atmosphere is dead once the dynamics have
        # advanced it; only its surface and physics namespace carry into
        # the second half (memory only: one spectral state, 1.19 GiB at
        # T533, no longer sits beside the second physics call).
        first_surface, first_physics_state = first.surface, first.physics_state
        del first
        with prof.section("physics_second"):
            second, second_physics = self._apply_physics_half(
                ArwenGlobalState(
                    advanced, first_surface, first_physics_state
                ), float(dt_s) if merged else half
            )
        if mark is not None:
            mark("physics_second", second.atmosphere)
        del advanced, first_surface, first_physics_state
        with prof.section("positivity_second"):
            repaired, negative_water, negative_tracer, second_fixer = (
                self._repair_positivity(second)
            )
        if mark is not None:
            mark("positivity_second", repaired.atmosphere)
        with prof.section("water_fixer"):
            repaired, water_offset = self._fix_total_water(repaired)
        repaired.atmosphere.step = bundle.step + 1
        repaired.atmosphere.time_s = bundle.time_s + float(dt_s)
        # The vapor fixer's magnitude this step: the global-mean column
        # water the four vapor clamps (two positivity repairs, two exchange
        # clamps) created and the column-local filler removed again, in
        # kg/m2 and relative to the atmosphere's global-mean column, plus
        # the largest fraction any column paid of its own vapor.  It is a
        # measured quantity of the truncated representation of vapor, not
        # a transfer: no reservoir and no other column is touched.
        fixer_kg_m2 = (
            first_fixer["water_kg_m2"] + second_fixer["water_kg_m2"]
            + float(first_physics.get("exchange_fixer_water_kg_m2", 0.0))
            + float(second_physics.get("exchange_fixer_water_kg_m2", 0.0))
        )
        fixer_max_rescale = max(
            first_fixer["max_rescale"], second_fixer["max_rescale"],
            float(first_physics.get("exchange_fixer_max_rescale", 0.0)),
            float(second_physics.get("exchange_fixer_max_rescale", 0.0)),
        )
        fixer_unfillable = (
            first_fixer["unfillable_kg_m2"] + second_fixer["unfillable_kg_m2"]
            + float(first_physics.get("exchange_fixer_unfillable_kg_m2", 0.0))
            + float(second_physics.get("exchange_fixer_unfillable_kg_m2", 0.0))
        )
        # Denominator: the largest global-mean column any of the four
        # clamps measured (they differ by one physics half-step; with the
        # positivity repair disabled only the exchange values exist).
        atmospheric_water = max(
            second_fixer["atmospheric_water_kg_m2"],
            first_fixer["atmospheric_water_kg_m2"],
            float(first_physics.get("exchange_atmospheric_water_kg_m2", 0.0)),
            float(second_physics.get("exchange_atmospheric_water_kg_m2", 0.0)),
        )
        metrics = {
            "spectral_cfl": float(cfl),
            "mass_fixer_log_offset": float(mass_offset),
            "global_water_fixer_kg_m2": float(water_offset),
            "positivity_fixer_water_kg_m2": float(fixer_kg_m2),
            "positivity_fixer_relative": float(
                fixer_kg_m2 / max(atmospheric_water, 1.0e-30)
            ),
            "positivity_fixer_max_rescale": float(fixer_max_rescale),
            "positivity_fixer_unfillable_kg_m2": float(fixer_unfillable),
            "maximum_repaired_negative_mixing_ratio": float(
                max(first_negative_water, negative_water)
            ),
            # The grid tracers are nonnegative by construction; this is
            # the roundoff floor's largest take, zero on a healthy step.
            "maximum_repaired_negative_number_per_kg": float(
                max(first_negative_tracer, negative_tracer)
            ),
            "tracer_transport_max_courant": float(max(
                transport["max_courant_x"], transport["max_courant_y"],
                transport["max_courant_z"],
            )),
            "tracer_transport_substeps": int(max(
                transport["substeps_x"], transport["substeps_y"],
                transport["substeps_z"],
            )),
            "tracer_transport_floor_clip_kg_m2": float(
                transport["floor_clip_kg_m2"]
            ),
            "tracer_transport_pseudo_density_mismatch_relative": float(
                transport["pseudo_density_mismatch_relative"]
            ),
            "tracer_transport": transport,
            **semi,
            "first_half_physics": first_physics,
            "second_half_physics": second_physics,
        }
        # The observer sees the state BEFORE the research-bound refusals so
        # a refused state has a ledger row and a snapshot, not just a
        # traceback.
        if self.observer is not None:
            with prof.section("observer"):
                self.observer.after_step(self, repaired, metrics)
        with prof.section("enforce"):
            self.enforce(repaired)
        if self.host_tier is not None:
            # Close the tier's step: retire the write-backs still in
            # flight (which releases the device buffers they read) and
            # read the transfer clock the receipt reports against the step
            # time (gate OVERLAP-1).
            self.host_tier.step_boundary()
        return repaired, metrics

    def diagnostics(self, bundle: ArwenGlobalState) -> dict[str, float]:
        """The run's per-output-step readings, taken a latitude band at a time.

        This is the widest grid state the run ever builds (the full
        dependency closure, fifteen fields), and it is built on output
        steps only.  Every reading below is either a plane the band fills
        or a minimum, maximum or sum over a band's own rows, so the
        levelled volumes exist at the band's width and the numbers are the
        whole grid's: the maxima and minima are exactly associative, and
        the two means run once over a finished plane.
        """
        xp = self.transform.backend.xp
        host = self.transform.backend.to_numpy
        grid = self.transform.grid
        sources = self.grid_sources(bundle.atmosphere)
        atmospheric = PlaneAccumulator(
            xp, grid.shape, self.transform.backend.float_dtype,
            name="diagnostics_atmospheric_water", exchange=self.pipeline.exchange,
        )
        speed_max = t_min = t_max = None
        condensate_max = moment_max = None
        for rows in self.pipeline.local_slices():
            g, _stack, _names = self.grid_band(sources, rows)
            atmospheric.add_band(
                rows, xp.sum(atmospheric_water_column(g), axis=0)
            )
            u = host(g["u"])
            v = host(g["v"])
            t = host(g["temperature"])
            speed = np.sqrt(u * u + v * v)
            speed_max = float(speed.max()) if speed_max is None else max(
                speed_max, float(speed.max()))
            t_min = float(t.min()) if t_min is None else min(t_min, float(t.min()))
            t_max = float(t.max()) if t_max is None else max(t_max, float(t.max()))
            del u, v, t, speed
            band_condensate = float(np.max(
                sum(host(g[name]) for name in ("qc", "qr", "qi", "qs", "qg"))
            ))
            condensate_max = band_condensate if condensate_max is None else max(
                condensate_max, band_condensate)
            band_moment = max(
                float(np.max(host(g[name]))) for name in NUMBER_MOMENTS
            )
            moment_max = band_moment if moment_max is None else max(
                moment_max, band_moment)
            del g
        ps = host(sources.pressure["ps"])
        stores = total_water_column(
            bundle, None, xp, atmospheric=atmospheric.plane
        )
        del sources, atmospheric
        atmospheric_water = host(stores["atmospheric"])
        surface_water = host(stores["surface"])
        soil_water = host(stores["soil"])
        native_water = host(stores["native"])
        outflow_water = host(stores["outflow"])
        total_water = host(stores["total"])
        return {
            "time_s": float(bundle.time_s),
            "step": int(bundle.step),
            "global_mean_surface_pressure_pa": float(grid.global_mean(ps)),
            "global_mean_atmospheric_water_kg_m2": float(
                grid.global_mean(atmospheric_water)
            ),
            "global_mean_surface_water_kg_m2": float(grid.global_mean(surface_water)),
            "global_mean_soil_water_kg_m2": float(grid.global_mean(soil_water)),
            "global_mean_native_water_kg_m2": float(grid.global_mean(native_water)),
            # Cumulative booked exits (runoff leaving the column system);
            # per-column values live in physics_state["water_outflow_kg_m2"].
            # The total below is held water PLUS this term, so the drift
            # gate and fixer target treat outflow as a booked exit.
            "global_mean_water_outflow_kg_m2": float(
                grid.global_mean(outflow_water)
            ),
            "global_mean_total_water_kg_m2": float(grid.global_mean(total_water)),
            "maximum_wind_m_s": float(speed_max),
            "minimum_temperature_k": float(t_min),
            "maximum_temperature_k": float(t_max),
            "minimum_surface_pressure_pa": float(ps.min()),
            "maximum_surface_pressure_pa": float(ps.max()),
            "maximum_total_condensate_kg_kg": float(condensate_max),
            "maximum_number_moment_per_kg": float(moment_max),
            "physics_state_array_count": int(len(bundle.physics_state.arrays)),
        }
