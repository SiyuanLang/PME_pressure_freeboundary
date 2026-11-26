"""Prescribed-data time-slab pressure representation."""
from dataclasses import asdict

import numpy as np
from scipy.special import roots_legendre

from solver import Config, InvalidGeometry
from standard_solver import StandardPressureMassELM


class TimeSlabPressureELM(StandardPressureMassELM):
    def __init__(self, m, config=None, precision_digits=50, minimum_degree=112):
        super().__init__(m, config, precision_digits, minimum_degree)
        self.time_origin = 0.
        self.slab_length = 0.
        self.initial_positions = np.array([-self.s0, self.s0])
        self.initial_y = roots_legendre(self.config.initial_points)[0]
        self.initial_pressure = self.Cm * (1 - self.initial_y**2)
        self.initial_source = 'prescribed_t0_datum'

    def boundary(self, t, jacobian=False):
        # Same stable sine basis, velocities and analytic Jac as the parent;
        # the two initial positions are now frozen numerical interface data.
        self._prepare()
        t = np.atleast_1d(np.asarray(t, dtype=float))
        key = t.tobytes()
        if key not in self._boundary_basis_cache:
            values, derivatives = [], []
            for ti in t:
                tt = self._mp_float(ti)
                values.append([[2*self.mp.sin(w*tt/2)*self.mp.cos(b+w*tt/2)
                                for w, b in zip(self._ws[side], self._bs[side])]
                               for side in range(2)])
                derivatives.append([[w*self.mp.cos(w*tt+b)
                                     for w, b in zip(self._ws[side], self._bs[side])]
                                    for side in range(2)])
            self._cache_put(self._boundary_basis_cache, key, (values, derivatives))
        values, derivatives = self._boundary_basis_cache[key]
        coefficient = [[self._mp_float(a) for a in row]
                       for row in self.params[self.Nv:].reshape(2, self.Ns)]
        s, sd = np.empty((t.size, 2)), np.empty((t.size, 2))
        c, h, cd, hd = (np.empty(t.size) for _ in range(4))
        for i in range(t.size):
            positions = [self.mp.fdot(values[i][side], coefficient[side])
                         + self._mp_float(self.initial_positions[side]) for side in range(2)]
            velocities = [self.mp.fdot(derivatives[i][side], coefficient[side])
                          for side in range(2)]
            s[i], sd[i] = [float(v) for v in positions], [float(v) for v in velocities]
            c[i], h[i] = float(sum(positions)/2), float((positions[1]-positions[0])/2)
            cd[i], hd[i] = float(sum(velocities)/2), float((velocities[1]-velocities[0])/2)
        if np.any(h <= 0) or not np.all(np.isfinite(h)):
            raise InvalidGeometry('The two learned boundaries must satisfy s1 < s2.')
        result = dict(t=t, s=s, sd=sd, c=c, h=h, cd=cd, hd=hd)
        if jacobian:
            js, jsd = np.zeros((t.size, 2, self.nparams)), np.zeros((t.size, 2, self.nparams))
            for side in range(2):
                sl = slice(self.Nv+side*self.Ns, self.Nv+(side+1)*self.Ns)
                js[:, side, sl] = [[float(v) for v in row[side]] for row in values]
                jsd[:, side, sl] = [[float(v) for v in row[side]] for row in derivatives]
            result.update(js=js, jsd=jsd, jc=js.mean(axis=1),
                jh=.5*(js[:, 1]-js[:, 0]), jcd=jsd.mean(axis=1),
                jhd=.5*(jsd[:, 1]-jsd[:, 0]))
        return result

    def _initial_pressure_target(self, y, geom):
        # Anchoring gives jx=0 at tau=0. Previous-slab coefficients are frozen,
        # so this target has no derivative with respect to CURRENT parameters.
        if not np.array_equal(np.asarray(y), self.initial_y):
            raise ValueError('Use the frozen initial-condition Gaussian grid.')
        return self.initial_pressure

    def set_numerical_initial_data(self, previous, previous_path):
        if previous.m != self.m or previous.M0 != self.M0:
            raise ValueError('The original prescribed mass must be inherited unchanged.')
        previous_t = previous.slab_length
        self.initial_positions = previous.boundary([previous_t])['s'][0].copy()
        g0 = self.boundary([0.])
        c, h = g0['c'][0], g0['h'][0]
        x = c+h*self.initial_y
        p, u = previous.predict(x, [previous_t])
        # Density is the inherited datum; its pressure is the positive part of
        # the preceding normalized pressure. No new mass renormalization occurs.
        self.initial_pressure = np.maximum(p[0], 0.)
        self.initial_source = json.dumps(dict(model_path=str(Path(previous_path).resolve()),
            previous_local_time=previous_t, physical_time=self.time_origin,
            density_transfer='previous_positive_part_density', mass_target='original_t0_M0'))
        assert np.allclose(self.initial_pressure, self.Cm*u[0]**self.k,
                           rtol=8e-15, atol=1e-15)

    def initialize_pressure(self, end_time):
        # Identical stationary-extension initialization, using the frozen datum.
        t = np.linspace(0., end_time, self.config.time_points)
        g0 = self.boundary([0.])
        c, h = g0['c'][0], g0['h'][0]
        x = c+h*self.initial_y
        phase = (x[None, :, None]*self.W_v[None, None, :, 0]
                 + t[:, None, None]*self.W_v[None, None, :, 1]+self.b_v)
        basis = np.sin(phase).reshape(-1, self.Nv)
        target = np.broadcast_to(self.initial_pressure, (t.size, x.size)).ravel()
        a, _, rank, _ = np.linalg.lstsq(basis, target, rcond=1e-12)
        self.params[:self.Nv], self.params[self.Nv:] = a, 0.
        norm = self.normalization(self.boundary(t))
        return dict(initialization='stationary_extension_of_current_frozen_initial_datum',
            initial_fit_rank=int(rank), initial_fit_rcond=1e-12,
            initial_fit_rms=float(np.sqrt(np.mean((basis@a-target)**2))),
            initial_raw_mass_min=float(np.exp(norm['log_m_raw']).min()),
            future_exact_data_used=False, extra_training_residuals=False)

    def state(self):
        return dict(m=self.m, params=self.params, W_v=self.W_v, b_v=self.b_v,
            W_s=self.W_s, b_s=self.b_s, M0=self.M0, s0=self.s0,
            q_nodes=self.q_nodes, q_weights=self.q_weights,
            time_origin=self.time_origin, slab_length=self.slab_length,
            initial_positions=self.initial_positions, initial_y=self.initial_y,
            initial_pressure=self.initial_pressure, initial_source=self.initial_source,
            pressure_head=self.pressure_head, density_convention='positive_part',
            problem_family='time_slab_standard_pressure',
            model_loader='time_slab_solver.TimeSlabPressureELM.load',
            stable_precision_digits=self.precision_digits,
            stable_minimum_degree=self.minimum_degree,
            **{f'config_{k}': v for k, v in asdict(self.config).items()})

    def save(self, path):
        np.savez_compressed(path, **self.state())

    @classmethod
    def from_state(cls, data):
        scalar = lambda k: np.asarray(data[k]).item()
        config = Config(**{k: scalar('config_'+k) for k in asdict(Config())})
        model = cls(scalar('m'), config, scalar('stable_precision_digits'),
                    scalar('stable_minimum_degree'))
        for k in ('params', 'W_v', 'b_v', 'W_s', 'b_s', 'q_nodes', 'q_weights',
                  'initial_positions', 'initial_y', 'initial_pressure'):
            setattr(model, k, np.array(data[k], copy=True))
        for k in ('M0', 's0', 'time_origin', 'slab_length', 'initial_source'):
            setattr(model, k, scalar(k))
        return model

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls.from_state(data)
