"""Resident output diagnostics, using the existing MPAS CUDA implementations.

MPAS v8.4.2 mpas_atm_core.F:675-701,825-910 schedules update/compute/write/reset.
No stream-manager operation or mutable atmospheric field crosses the host.
"""
from woof.hex.device_memory_release import shared_card_memory
from datetime import timedelta

from ..mpasdiag.isobaric import FIELDS as ISOBARIC, IsobaricDiagnostics
from ..mpasdiag.convective_v842 import FIELDS as CONVECTIVE, MAXIMA, ConvectiveDiagnosticsModule
from ..mpasdiag.utils import DiagnosticsUtils, Direction, StreamView
from ..mpasdiag import nwp_diag_v842 as _nwp
from ..mpasdiag import tslist_v842 as _tslist
from ..mpasdiag import plevel_diag_v842 as _plevel
from ..mpasdiag import acc_diag_v842 as _acc
from ..mpasdiag import afwa_diag_v842 as _afwa
from ..mpasdiag import lightning_diag_v842 as _lightning
from ..mpasdiag import hailcast_diag_v842 as _hailcast
from ..mpasdiag import reduction_diag_v842 as _reduce


PRIMARY = frozenset(('vorticity', 'divergence', 'ke'))
# Registry.xml:606 reads theta_base at initialization; :1853 circulation has
# no dynamics writer. Preserve those initialized words, including restart
# input, rather than manufacturing a zero or dividing rounded base products.
INITIALIZED = frozenset(('circulation', 'theta_base'))
MOMENTUM = {'ru': 'rho_u', 'rw': 'rho_w'}
ACOUSTIC = frozenset(('ru_p', 'rw_p'))
PRODUCED = (PRIMARY | frozenset(ISOBARIC) | frozenset(CONVECTIVE)
            | {'t_iso_levels', 'z_iso_levels'} | INITIALIZED | MOMENTUM.keys() | ACOUSTIC)


class _Pools:
    def __init__(self, owner, pool):
        self.owner, self.pool = owner, pool

    def __getitem__(self, name):
        owner, pool = self.owner, self.pool
        if pool == 'mesh' and name in owner.session.layout.dimensions:
            return owner.session.layout.dimensions[name]
        if pool == 'state' and name == 'index_qv':
            return owner.session.layout.indices[pool, name]
        if pool == 'diag' and name in owner.produced:
            return owner.buffer(pool, name)
        return owner.current(pool, name)

    def padded_qv(self):
        """state scalars(index_qv,:,:) at native padded extent: the words ``self['scalars'][index_qv-1]`` gives.

        THE BREAKAGE THIS PREVENTS (vram census 2026-10-28): isobaric.py reads only the moisture slot, but
        ``self['scalars']`` kept a padded copy of every scalar for the whole run (3,068 B per column on hrrrv5).
        """
        if self.pool != 'state':
            raise KeyError('padded_qv is a state-pool view')
        return self.owner.current_qv()

    def view(self, name):
        """The current authority's own words for a kernel that reads only the owned extent in place.

        On the resident route that is the device allocation itself, without the padded copy __getitem__ makes
        (the padding words past the owned cells are never read through a view). Everything else, and every
        non-resident pool, is __getitem__.
        """
        owner, pool = self.owner, self.pool
        if (owner.session.device_pools.resident
                and not (pool == 'mesh' and name in owner.session.layout.dimensions)
                and not (pool == 'state' and name == 'index_qv')
                and not (pool == 'diag' and name in owner.produced)):
            return owner.authority(pool, name)
        return self[name]

    def __contains__(self, name):
        # mpas_pool_get_array's association test: an allocated field of this pool.
        layout = self.owner.session.layout
        return ((self.pool == 'mesh' and name in layout.dimensions)
                or any(row['pool'] == self.pool and row['pool_name'] == name for row in layout.fields))


class ResidentOutputDiagnostics:
    """Keep produced fields on the device until the writer takes a snapshot.

    Closed single-rank meshes need no pressure or state halo exchange.
    Temporary padded views preserve the input allocation's sentinel words;
    authoritative resident pool allocations keep their existing owned extent.
    """
    produced = PRODUCED
    #: WRF's nwp_diagnostics maxima (mpasdiag.nwp_diag_v842); built by bind() only with config_nwp_diagnostics.
    nwp = None
    #: WRF's per-step station time series (mpasdiag.tslist_v842); built by bind() only with config_tslist.
    tslist = None
    #: WRF's p_lev / z_lev diagnostics (mpasdiag.plevel_diag_v842); built by bind() only with config_p_lev_diags / config_z_lev_diags.
    plevel = None
    #: WRF's prec_acc_dt windows / acc_phy_tend accumulators (mpasdiag.acc_diag_v842); built by bind() only with
    #: config_prec_acc_dt / config_acc_phy_tend.
    accdiag = None
    #: WRF's AFWA diagnostics (mpasdiag.afwa_diag_v842); built by bind() only with a config_afwa_*_opt option.
    afwa = None
    #: WRF's lightning diagnostics (mpasdiag.lightning_diag_v842); built by bind() only with config_lightning_option.
    lightning = None
    #: WRF-HAILCAST (mpasdiag.hailcast_diag_v842); built by bind() only with config_hailcast_opt.
    hailcast = None
    #: WRF's CLWRF / RASM / surface-energy reductions (mpasdiag.reduction_diag_v842); built by bind() only with their options.
    reduction = None
    #: The running maxima/accumulators of the convective module (``accumulates``).
    convective_maxima = MAXIMA
    #: Pools the diagnostic modules read.
    pool_names = ('mesh', 'state', 'diag')
    #: The straight run's outer steps before this segment (runtime_v842 sets it from the checkpoint route's
    #: physics_clock_offset on a correct-mode continuation, else 0). The loop's steps stay local (output naming,
    #: the step-0 resume); the modules on WRF's elapsed-time clock (lightning) add it, as physics does.
    step_offset = 0

    def __init__(self, session, dycore, *, halo=None):
        self.session, self.dycore = session, dycore
        # None on a closed single-rank mesh. A same-host rank's owner fills
        # the halo cells the vertex pressure (isobaric.py) averages over.
        self.halo = halo
        self.cp = session.device_pools.cp
        self.pools = {name: _Pools(self, name) for name in self.pool_names}
        self.buffers = {}
        self.writer = None
        self.when = None
        self.updates = 0
        self.last_step = None
        self.acoustic = {}
        self.acoustic_captures = 0
        self.pending = None
        self.deferred_updates = 0
        # One output event's stream views (before_write .. after_snapshot): see views().
        self._event_views = None
        self._fields, self._fields_state = {}, None
        # A checkpoint continuation's running accumulators (resume_accumulators), installed at step 0.
        self._resume = None
        # The optional modules' fields, known from the active packages before bind(): the
        # output writer (streams_output_v842.PhysicsStreams) checks every enabled stream's
        # fields against ``produced`` when it is built, which is before it binds this
        # object.  THE BREAKAGE: with produced extended only in bind(), every live run
        # with config_nwp_diagnostics, p/z_lev, prec_acc/acc_phy_tend, AFWA or the
        # CLWRF/mean/diurnal/surface-energy reductions stopped before step 1 with
        # "MPAS output fields lack computed authority".
        self.produced = type(self).produced | self._package_fields()

    def _package_fields(self):
        """The diagnostic fields the active Registry packages make this object produce."""
        packages = set(getattr(self.session.layout, 'packages', ()) or ())
        fields = set()
        if _nwp.PACKAGE in packages:
            fields |= set(_nwp.NAMES)
        if {_acc.PREC_PACKAGE, _acc.TEND_PACKAGE} & packages:
            fields |= set(_acc.NAMES)
        if set(_reduce.PACKAGES) & packages:
            fields |= set(_reduce.NAMES)
        if set(_afwa.SELECTORS) & packages:
            fields |= set(_afwa.NAMES)
        if set(_lightning.PACKAGES.values()) & packages:
            fields |= set(_lightning.NAMES)
        if _hailcast.PACKAGE in packages:
            fields |= set(_hailcast.NAMES)
        if {_plevel.P_PACKAGE, _plevel.Z_PACKAGE} & packages:
            fields |= set(_plevel.NAMES)
        return frozenset(fields)

    def accumulator_state(self):
        """The convective running accumulators' words, for a checkpoint that a continuation resumes."""
        state = {name: self.cp.asnumpy(self.buffers['diag', name])
                 for name in self.convective_maxima if ('diag', name) in self.buffers}
        if self.nwp is not None:
            state.update(self.nwp.state())
        if self.accdiag is not None:
            state.update(self.accdiag.state())
        if self.afwa is not None:
            state.update(self.afwa.state())
        if self.lightning is not None:
            state.update(self.lightning.state())
        if self.hailcast is not None:
            state.update(self.hailcast.state())
        if self.reduction is not None:
            state.update(self.reduction.state())
        return state

    def resume_accumulators(self, state):
        """Continue a checkpoint's running accumulators instead of native's start-of-run update.

        Native starts every run with reset, update, compute, write, reset (fork mpas_atm_core.F:672-715); the resets
        clear only fields a due stream writes, and a restart's output/diagnostics alarms were reset, so a restart
        folds its start instant into the accumulators a second time.  THE BREAKAGE THIS FIXES: a continuation from
        a checkpoint on the diagnostics frame summed six samples into the next frame's means where the straight
        run summed five (hrrrv5 x1.2562, 14Z diag: u10m_mean, prec_acc_nc and every updraft-helicity maximum,
        28,342 words).  The straight run already folded the checkpoint instant before it sealed.
        """
        if self.tslist is not None:
            raise ValueError("a checkpoint continuation of a run with &diagnostics config_tslist output is not carried: "
                             "the Rust station writer's resume receipt is not in hex's checkpoint, so the station "
                             "files would restart from empty")
        self._resume = {name: value for name, value in state.items()
                        if name in self.convective_maxima and ('diag', name) in self.buffers}
        if self.nwp is not None:
            self._resume.update((name, value) for name, value in state.items()
                                if name.startswith(_nwp.STATE_PREFIX))
        if self.accdiag is not None:
            self._resume.update((name, value) for name, value in state.items()
                                if name.startswith(_acc.STATE_PREFIX))
        if self.afwa is not None:
            self._resume.update((name, value) for name, value in state.items()
                                if name.startswith(_afwa.STATE_PREFIX))
        if self.lightning is not None:
            self._resume.update((name, value) for name, value in state.items()
                                if name.startswith(_lightning.STATE_PREFIX))
        if self.hailcast is not None:
            self._resume.update((name, value) for name, value in state.items()
                                if name.startswith(_hailcast.STATE_PREFIX))
        if self.reduction is not None:
            self._resume.update((name, value) for name, value in state.items()
                                if name.startswith(_reduce.STATE_PREFIX))

    def buffer(self, pool, name):
        key = pool, name
        if key not in self.buffers:
            # Initial allocation/default/dummy words are read once. All later
            # changes come from resident views or the diagnostic CUDA kernels.
            host = self.session.native.get_field(pool, name, level=1)
            self.buffers[key] = self.cp.asarray(host, order='C').reshape(host.shape or (1,))
        return self.buffers[key]

    def authority(self, pool, name):
        """The current level-1 authority of a pool field (resident device allocation, else the host pool's)."""
        p = self.session.device_pools
        old = p.level
        try:
            p.level = 1
            # Nonresident coupling updates host pools directly. Cached slot
            # views can predate those writes, including when no physics slot
            # runs to clear them. Fetch the current authority for each input.
            return (p.field(pool, name) if p.resident else self.cp.asarray(
                    self.session.native.get_field(pool, name, level=1)))
        finally:
            p.level = old

    def current(self, pool, name):
        cp = self.cp
        target = self.buffer(pool, name)
        value = self.authority(pool, name)
        target[tuple(slice(0, size) for size in value.shape)] = value
        dims = self.session.layout.dimensions
        if pool == 'mesh' and name in ('cellsOnVertex', 'kiteAreasOnVertex'):
            return cp.ascontiguousarray(target[:, :dims['nVertices']])
        if pool == 'mesh' and name == 'areaTriangle':
            return cp.ascontiguousarray(target[:dims['nVertices']])
        return target

    def current_qv(self):
        """current('state', 'scalars')[index_qv-1] without the other scalars' padded copy (same words).

        The buffer starts from the initial level-1 moisture slot (its padding words included, as buffer() reads
        the whole array once) and takes the authority's leading words on every call, as current() does.
        """
        cp = self.cp
        index = int(self.session.layout.indices['state', 'index_qv']) - 1
        key = 'state', 'scalars:index_qv'
        if key not in self.buffers:
            initial = self.session.native.get_field('state', 'scalars', level=1)
            self.buffers[key] = cp.array(initial[index], dtype=initial.dtype, copy=True, order='C')
        target = self.buffers[key]
        value = self.authority('state', 'scalars')[index]
        target[tuple(slice(0, size) for size in value.shape)] = value
        return target

    def bind(self, writer):
        self.writer = writer
        self.when = writer.current_time(step=0, dt=0)
        self.utils = DiagnosticsUtils()
        self.utils.init(self.views)
        self.isobaric = IsobaricDiagnostics(exchange_pressure_halo=(lambda p: None) if self.halo is None
                                            else self.halo.exchange_pressure)
        self.isobaric.setup(self.pools, getattr(self.session.native, 'clock_state', None), self.utils)
        self.convective = self._new_convective()
        self.convective.setup(self._convective_pools(), getattr(self.session.native, 'clock_state', None), self.utils)
        if self.halo is not None:
            self.convective.driver.vertex_order = self.halo.vertex_order()
        self.accumulates = any(self.utils.stream_inclusion_count(name, Direction.OUTPUT)
                               for name in self.convective_maxima)
        if _nwp.PACKAGE in self.session.layout.packages:
            self.nwp = _nwp.NwpDiagnosticsModule(self)
            self.produced = PRODUCED | frozenset(_nwp.NAMES)
            self.nwp.setup(self.utils)
            self.accumulates = self.accumulates or self.nwp.accumulates
        stations = getattr(self.session.config, 'tslist', None)
        if stations is not None:
            self.tslist = _tslist.TslistModule(self, stations)
            self.tslist.setup()
            self.accumulates = True
        if {_acc.PREC_PACKAGE, _acc.TEND_PACKAGE} & set(self.session.layout.packages):
            self.accdiag = _acc.AccumulatorModule(self, self.session.config.acc_diagnostics)
            self.produced = self.produced | frozenset(_acc.NAMES)
            self.accdiag.setup(self.utils, self.dycore.config.config_dt)
            self.accumulates = self.accumulates or self.accdiag.accumulates
        if set(_reduce.PACKAGES) & set(self.session.layout.packages):
            self.reduction = _reduce.ReductionModule(self, self.session.config.reduction_diagnostics)
            self.produced = self.produced | frozenset(_reduce.NAMES)
            self.reduction.setup(self.utils, self.dycore.config.config_dt)
            self.accumulates = self.accumulates or self.reduction.accumulates
        if set(_afwa.SELECTORS) & set(self.session.layout.packages):
            self.afwa = _afwa.AfwaModule(self, self.session.config.afwa_diagnostics)
            self.produced = self.produced | frozenset(_afwa.NAMES)
            self.afwa.setup(self.utils)
            self.accumulates = self.accumulates or self.afwa.accumulates
        if set(_lightning.PACKAGES.values()) & set(self.session.layout.packages):
            self.lightning = _lightning.LightningModule(self, self.session.config.lightning_diagnostics)
            self.produced = self.produced | frozenset(_lightning.NAMES)
            self.lightning.setup(self.utils)
            self.accumulates = True
        if _hailcast.PACKAGE in self.session.layout.packages:
            self.hailcast = _hailcast.HailcastModule(self, self.session.config.hailcast_diagnostics)
            self.produced = self.produced | frozenset(_hailcast.NAMES)
            self.hailcast.setup(self.utils)
            self.accumulates = True
        if {_plevel.P_PACKAGE, _plevel.Z_PACKAGE} & set(self.session.layout.packages):
            self.plevel = _plevel.LevelDiagnosticsModule(self, self.session.config.plevel_diagnostics)
            self.produced = self.produced | frozenset(_plevel.NAMES)
            self.plevel.setup(self.utils)

    def _new_convective(self):
        """The route's convective diagnostics module (MPAS v8.4.2's here)."""
        return ConvectiveDiagnosticsModule()

    def _convective_pools(self):
        """The pool views the convective module reads (the shared ones here)."""
        return self.pools

    def _update_convective(self):
        """mpas_atm_core.F:675-701 mpas_atm_diag_update, with the arguments the route's module takes."""
        self.convective.update()
        if self.nwp is not None:
            self.nwp.update()
        if self.accdiag is not None:
            self.accdiag.update()
        if self.afwa is not None:
            self.afwa.update()
        if self.lightning is not None:
            self.lightning.update()
        if self.hailcast is not None:
            self.hailcast.update()
        if self.reduction is not None:
            self.reduction.update()
        if self.tslist is not None:
            self.tslist.update()

    def views(self):
        """The writer's streams as MPAS stream views at ``self.when`` (a tuple).

        Within one output event (before_write until after_snapshot) every query sees the same alarms, so the
        views are built once and the same tuple is returned (DiagnosticsUtils then answers every
        will-be-written query from one name set). Outside an event every call builds them afresh. A stream's
        field list is a pure function of the stream and the active packages, so it is expanded once per run.
        """
        if self._event_views is not None:
            return self._event_views
        packages, streams = self.session.layout.packages, self.writer.streams
        state = (frozenset(packages), tuple(streams.values()))
        held = self._fields_state
        if (held is None or held[0] != state[0] or len(held[1]) != len(state[1])
                or any(a is not b for a, b in zip(held[1], state[1]))):
            # Expansion reads the whole stream set (included streams), so any change re-expands every stream.
            self._fields, self._fields_state = {}, state
        views = []
        for name, stream in streams.items():
            fields = self._fields.get(name)
            if fields is None:
                fields = self._fields[name] = stream.fields(packages, streams=streams)
            directions = stream.options['type'].split(';')
            output = 'output' in directions
            views.append(StreamView(name,
                Direction.INPUT_OUTPUT if len(directions) == 2 else
                    Direction.OUTPUT if output else Direction.INPUT,
                bool(fields),
                output and self.writer.alarms.due(name, self.when),
                dict.fromkeys(fields, True),
                stream.interval('input'), stream.interval('output')))
        return tuple(views)

    def _begin_event(self):
        # alarms.due is evaluated once per stream for the event, at self.when.
        self._event_views = None
        self._event_views = self.views()

    def _end_event(self):
        self._event_views = None

    def defer(self, *, step, when):
        """Hold step's running-maximum update for the next synchronize.

        mpas_atm_core.F:675-701 updates the maxima after every step from the
        state that step left. When no stream is due, that state is unchanged
        until the next step's outer physics synchronizes it into the pools
        (winds, w, cell-reconstructed winds); flush() runs the update there,
        on the same words, so no publication or host sync is forced per step.
        """
        if self.accumulates:
            if self.pending is not None:
                raise RuntimeError(f'maxima of step {self.pending[0]} were never folded')
            self.pending = (step, when)

    def flush(self):
        """Fold the deferred step's maxima; call right after synchronize."""
        if self.pending is None:
            return
        step, when = self.pending
        self.pending = None
        self.when = when
        if self.tslist is not None:
            self.tslist.begin(step, when)
        if self.accdiag is not None:
            self.accdiag.begin(step, when)
        if self.afwa is not None:
            self.afwa.begin(step, when)
        if self.lightning is not None:
            self.lightning.begin(step, when)
        if self.hailcast is not None:
            self.hailcast.begin(step, when)
        if self.reduction is not None:
            self.reduction.begin(step, when)
        # No stream is due on a deferred step, so only the maxima read the
        # primary diagnostics: vorticity alone (divergence/KE are formed
        # again by before_write whenever a stream needs them).
        a, horizontal = self.dycore.atmosphere, self.dycore.horizontal
        if hasattr(horizontal, 'solve_vorticity'):
            value = horizontal.solve_vorticity(a.state.rho, a.state.rho_u,
                                               normal_velocity=a.saved.normal_velocity)
            target = self.buffer('diag', 'vorticity')
            target[..., :value.shape[-1]] = value
        else:
            self._solve_primary()
        self._update_convective()
        self.updates += 1
        self.deferred_updates += 1
        self.last_step = step

    def _solve_primary(self):
        # Final solve vorticity/KE/divergence depend only on normal velocity,
        # unchanged by subsequent scalar transport and microphysics. Reuse the
        # existing dycore operator, including native reduction order.
        a, driver = self.dycore.atmosphere, self.dycore
        solved = driver.horizontal.solve_diagnostics(a.state.rho, a.state.rho_u,
            dt=driver.config.config_dt, apvm_upwinding=driver.config.config_apvm_upwinding,
            normal_velocity=a.saved.normal_velocity, rk_step=3)
        for name, value in (('vorticity', solved.vorticity),
                            ('divergence', solved.divergence), ('ke', solved.kinetic_energy)):
            target = self.buffer('diag', name)
            target[..., :value.shape[-1]] = value

    def before_write(self, *, step, when):
        if self.pending is not None:
            # A deferred update must be folded on its own state before this one.
            raise RuntimeError(f'maxima of step {self.pending[0]} were not folded before step {step}')
        self.when = when
        if self.tslist is not None:
            self.tslist.begin(step, when)
        if self.accdiag is not None:
            self.accdiag.begin(step, when)
        if self.afwa is not None:
            self.afwa.begin(step, when)
        if self.lightning is not None:
            self.lightning.begin(step, when)
        if self.hailcast is not None:
            self.hailcast.begin(step, when)
        if self.reduction is not None:
            self.reduction.begin(step, when)
        self._begin_event()
        resume, self._resume = (self._resume, None) if step == 0 else (None, self._resume)
        if resume is not None:
            for name, value in resume.items():
                if not name.startswith((_nwp.STATE_PREFIX, _acc.STATE_PREFIX, _afwa.STATE_PREFIX, _lightning.STATE_PREFIX,
                                         _hailcast.STATE_PREFIX, _reduce.STATE_PREFIX)):
                    target = self.buffers['diag', name]
                    target[...] = self.cp.asarray(value).reshape(target.shape)
            if self.nwp is not None:
                self.nwp.resume(resume)
            if self.accdiag is not None:
                self.accdiag.resume(resume)
            if self.afwa is not None:
                self.afwa.resume(resume)
            if self.lightning is not None:
                self.lightning.resume(resume)
            if self.hailcast is not None:
                self.hailcast.resume(resume)
            if self.reduction is not None:
                self.reduction.resume(resume)
        elif step == 0:
            self.convective.reset()
            if self.nwp is not None:
                self.nwp.reset()
        if self.accumulates or any(view.output_alarm for view in self.views()):
            self._solve_primary()
        if resume is None:
            self._update_convective()
        self.isobaric.compute()
        self.convective.compute()
        if self.nwp is not None:
            self.nwp.compute()
        if self.accdiag is not None:
            self.accdiag.compute()
        if self.afwa is not None:
            self.afwa.compute()
        if self.lightning is not None:
            self.lightning.compute()
        if self.hailcast is not None:
            self.hailcast.compute()
        if self.reduction is not None:
            self.reduction.compute()
        if self.plevel is not None:
            self.plevel.compute()
        self.updates += 1
        self.last_step = step

    def prepare_step(self, *, step, dt):
        """Retain acoustic allocations only for a due output, not ordinary steps."""
        # An event whose write found no due stream ends here (write() returns before after_snapshot).
        self._end_event()
        self.acoustic.clear()
        when = (self.writer.current_time(step=step-1, dt=dt)
                + timedelta(seconds=float(dt)))
        for name, stream in self.writer.streams.items():
            if ('output' in stream.options['type'].split(';') and
                    ACOUSTIC.intersection(stream.fields(self.session.layout.packages,
                                                       streams=self.writer.streams)) and
                    self.writer.alarms.due(name, when)):
                return self.capture_acoustic
        return None

    def capture_acoustic(self, *, ru_p, rw_p):
        """The last RK stage/subcycle supplies its original device words."""
        self.acoustic.update(ru_p=ru_p, rw_p=rw_p)
        self.acoustic_captures += 1

    def after_snapshot(self):
        try:
            self.convective.reset()
            if self.nwp is not None:
                self.nwp.reset()
            if self.afwa is not None:
                self.afwa.reset()
            if self.hailcast is not None:
                self.hailcast.reset()
        finally:
            self._end_event()
        self.acoustic.clear()
        if not shared_card_memory():
            return
        # These padded inputs are read only by output-time diagnostics. Maxima
        # retain their own fields and inputs. Rebuild owned words from current()
        # at the next output; the native allocation supplies unchanged padding.
        for key in (('state', 'scalars'), ('state', 'scalars:index_qv'), ('state', 'theta_m'),
                    ('diag', 'exner'), ('diag', 'pressure_base'), ('diag', 'pressure_p')):
            self.buffers.pop(key, None)
        # These products are computed afresh for every due frame. Neither the
        # running maxima nor initialized-only diagnostics use their previous
        # values. The host Registry allocation preserves the padding words.
        for name in PRIMARY | frozenset(ISOBARIC) | ACOUSTIC:
            self.buffers.pop(('diag', name), None)

    def field(self, pool, name):
        # atm_recover_large_step_variables_work F:4418,4476 writes rw/ru.
        # The CUDA prognostic state already owns those exact recovery words;
        # no velocity conversion or host pool publication is needed here.
        if pool == 'diag' and name in MOMENTUM:
            return getattr(self.dycore.atmosphere.state, MOMENTUM[name])
        if pool == 'diag' and name in ACOUSTIC:
            if self.last_step == 0:
                return self.buffer(pool, name)
            if name not in self.acoustic:
                raise RuntimeError(f'{name}: final acoustic output was not retained')
            return self.acoustic[name]
        if pool == 'diag' and name in self.produced:
            return self.buffer(pool, name)
        return None

    def close(self):
        if self.tslist is not None:
            self.tslist.close()
        self.acoustic.clear()
        self.buffers.clear()

    def receipt(self):
        return dict(updates=self.updates, last_step=self.last_step,
                    deferred_updates=self.deferred_updates,
                    produced=sorted(self.produced),
                    device_buffer_bytes=sum(value.nbytes for value in self.buffers.values()),
                    acoustic_captures=self.acoustic_captures,
                    acoustic_buffer_bytes=sum(value.nbytes for value in self.acoustic.values()),
                    pressure_halo='closed single-rank mesh' if self.halo is None
                                  else 'same-host rank owner exchange')
