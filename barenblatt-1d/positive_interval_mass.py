"""Shared moving-positive-interval mass layer from the root-fractional study.

Analytic derivatives require fixed zero topology and simple internal zeros.
Learned support endpoints use geometry derivatives, even when q_x vanishes.
A finite sign scan does not certify absence of tangential/narrow zero pairs.
The raw-jet hook lets stable arithmetic reuse exactly the same mass formulas.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_jacobi


MASS_RULE = 'moving_positive_intervals_fractional_gauss_jacobi'


class PositiveIntervalMassMixin:
    mass_rule = MASS_RULE

    def __init__(self, *args, order=64, scan=257, endpoint_weight='roots', **kwargs):
        super().__init__(*args, **kwargs)
        self.root_order, self.root_scan = int(order), int(scan)
        self.endpoint_weight = endpoint_weight
        if self.root_order < 2 or self.root_scan < 3:
            raise ValueError('Mass order >= 2 and root scan >= 3 are required.')
        if endpoint_weight not in ('roots', 'all'):
            raise ValueError('endpoint_weight must be roots or all.')
        self._rules = {}

    def _rule(self, left_root, right_root, order=None):
        if self.endpoint_weight == 'all':
            left_root = right_root = True
        order = self.root_order if order is None else int(order)
        key = (left_root, right_root, order)
        if key not in self._rules:
            alpha, beta = self.nu*right_root, self.nu*left_root
            y, w = roots_jacobi(order, alpha, beta)
            z = (y+1)/2
            # Divide out the Jacobi weight: the physical integrand is unchanged.
            w = w/2**(alpha+beta+1)/(z**beta*(1-z)**alpha)
            self._rules[key] = z, w
        return self._rules[key]

    def _raw(self, x, t):
        phase = np.asarray(x)[..., None]*self.W_v[:, 0]+t*self.W_v[:, 1]+self.b_v
        return np.sin(phase)@self.params[:self.Nv]

    def _mass_jet(self, x, t, jacobian=False):
        """q and physical derivatives; parameter derivatives hold x fixed."""
        phase = np.asarray(x)[..., None]*self.W_v[:, 0]+t*self.W_v[:, 1]+self.b_v
        sine, cosine = np.sin(phase), np.cos(phase)
        a, wx, wt = self.params[:self.Nv], self.W_v[:, 0], self.W_v[:, 1]
        result = dict(q=sine@a, qx=cosine@(wx*a), qt=cosine@(wt*a),
                      qxx=sine@(-wx*wx*a), qxt=sine@(-wx*wt*a))
        if jacobian:
            shape = np.shape(result['q'])+(self.nparams,)
            qj, qxj, qtj = (np.zeros(shape) for _ in range(3))
            qj[..., :self.Nv], qxj[..., :self.Nv], qtj[..., :self.Nv] = sine, cosine*wx, cosine*wt
            result.update(qj=qj, qxj=qxj, qtj=qtj)
        return result

    def _endpoint(self, x, t, root, geom, i, side, jacobian):
        if not root:
            return (x, geom['sd'][i, side],
                    geom['js'][i, side] if jacobian else None,
                    geom['jsd'][i, side] if jacobian else None)
        f = self._mass_jet(x, t, jacobian)
        qx = f['qx']
        if qx == 0 or not np.isfinite(qx):
            raise FloatingPointError('Degenerate internal root: implicit derivatives unavailable.')
        xt = -f['qt']/qx
        if not jacobian:
            return x, xt, None, None
        xj = -f['qj']/qx
        xtj = -(f['qtj']+f['qxt']*xj+(f['qxj']+f['qxx']*xj)*xt)/qx
        return x, xt, xj, xtj

    def _positive_cuts(self, xs, qs, t):
        roots = []
        for j in range(len(xs)-1):
            if qs[j] == 0 and j > 0:
                roots.append(float(xs[j]))
            elif (qs[j] > 0) != (qs[j+1] > 0) and qs[j+1] != 0:
                try:
                    roots.append(brentq(lambda x: float(self._raw(x, t)), xs[j], xs[j+1],
                                        xtol=5e-15, rtol=1e-14))
                except (ValueError, RuntimeError) as exc:
                    raise FloatingPointError('Unreliable positive-interval root bracket.') from exc
        # Endpoints belong to geometry, not to the implicit internal-root branch.
        roots = np.unique(roots)
        roots = roots[(roots > xs[0]) & (roots < xs[-1])]
        return np.r_[xs[0], roots, xs[-1]]

    def normalization(self, geom, jacobian=False):
        times = geom['t']
        y = np.linspace(-1., 1., self.root_scan)
        grid = geom['c'][:, None]+geom['h'][:, None]*y
        samples = self.shape_at(grid, times)
        masses, mts, mjs, mtjs, mass_samples, positive_intervals = [], [], [], [], [], []
        minimum = float('inf')
        root_counts = []
        for i, t in enumerate(times):
            cuts = self._positive_cuts(grid[i], samples[i], t)
            root_counts.append(len(cuts)-2)
            mass = mt = 0.
            intervals, interval_samples = [], []
            mj, mtj = np.zeros(self.nparams), np.zeros(self.nparams)
            for j in range(len(cuts)-1):
                if self._raw((cuts[j]+cuts[j+1])/2, t) <= 0:
                    continue
                lr, rr = j > 0, j+1 < len(cuts)-1
                l, lt, lj, ltj = self._endpoint(cuts[j], t, lr, geom, i, 0, jacobian)
                r, rt, rj, rtj = self._endpoint(cuts[j+1], t, rr, geom, i, 1, jacobian)
                length, length_t = r-l, rt-lt
                z, w = self._rule(lr, rr)
                x, xt = l+length*z, lt+length_t*z
                f = self._mass_jet(x, t, jacobian)
                q, qx, qt = f['q'], f['qx'], f['qt']
                if np.any(q <= 0) or not np.isfinite(q).all():
                    raise FloatingPointError('Unresolved positive interval or pressure arithmetic failure.')
                intervals.append((l, r, lr, rr))
                interval_samples.append((q, length*w))
                minimum = min(minimum, float(q.min()))
                g = qt+qx*xt
                Z, Zt = w@q**self.nu, self.nu*(w*q**(self.nu-1))@g
                mass += length*Z
                mt += length_t*Z+length*Zt
                if jacobian:
                    length_j, length_tj = rj-lj, rtj-ltj
                    xj = lj[None, :]+z[:, None]*length_j
                    xtj = ltj[None, :]+z[:, None]*length_tj
                    qj = qx[:, None]*xj+f['qj']
                    qtj = f['qxt'][:, None]*xj+f['qtj']
                    qxj = f['qxx'][:, None]*xj+f['qxj']
                    gj = qtj+qxj*xt[:, None]+qx[:, None]*xtj
                    Zj = self.nu*(w*q**(self.nu-1))@qj
                    Ztj = self.nu*((w*q**(self.nu-1))@gj
                        +(self.nu-1)*(w*q**(self.nu-2)*g)@qj)
                    mj += length_j*Z+length*Zj
                    mtj += length_tj*Z+length_t*Zj+length_j*Zt+length*Ztj
            if mass <= 0 or not np.isfinite(mass):
                raise FloatingPointError('No positive raw mass.')
            masses.append(mass*self.Cm**(-self.nu))
            mts.append(mt*self.Cm**(-self.nu))
            mjs.append(mj*self.Cm**(-self.nu))
            mtjs.append(mtj*self.Cm**(-self.nu))
            mass_samples.append(interval_samples)
            positive_intervals.append(intervals)
        M, Mt = np.array(masses), np.array(mts)
        result = dict(raw_mass=M, mass_samples=mass_samples, positive_intervals=positive_intervals,
                      internal_root_counts=np.array(root_counts),
                      log_m_raw=np.log(M), log_a=self.k*(np.log(self.M0)-np.log(M)),
                      log_a_t=-self.k*Mt/M,
                      log_i=np.log(M)+self.nu*np.log(self.Cm)-np.log(geom['h']),
                      minimum_raw_pressure_at_mass_nodes=minimum, negative_mass_node_count=0)
        if jacobian:
            Mj, Mtj = np.array(mjs), np.array(mtjs)
            result.update(jlog_a=-self.k*Mj/M[:, None],
                          jlog_a_t=-self.k*(Mtj/M[:, None]-Mt[:, None]*Mj/M[:, None]**2))
        return result

    def mass_curve(self, t, n_quad=None, rule='jacobi'):
        if n_quad is not None:
            raise ValueError('Report the training mass rule only.')
        norm = self.normalization(self.boundary(t))
        return np.array([sum(w@((np.exp(la)*q/self.Cm)**self.nu) for q, w in intervals)
                         for la, intervals in zip(norm['log_a'], norm['mass_samples'])])
