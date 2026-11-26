"""Mass-normalized pressure ELM with two independent moving boundaries.

The solver uses analytic residuals/Jacobians and numpy.linalg.lstsq.
Only the initial Barenblatt datum enters this module; future reference
solutions belong to the separate experiment/evaluation script.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
from scipy.special import gammaln, roots_jacobi


@dataclass
class Config:
    seed: int = 1234
    pressure_neurons: int = 192
    boundary_neurons: int = 24
    pressure_frequency_scale: float = 0.75
    boundary_frequency_scale: float = 1.0
    mass_quad_n: int = 64
    time_points: int = 40
    space_points: int = 32
    initial_points: int = 80
    gamma_darcy: float = 100.0
    gamma_initial: float = 100.0
    rcond: float = 1.0e-8
    max_iter: int = 40
    residual_tol: float = 1.0e-9
    step_tol: float = 1.0e-9
    max_backtracks: int = 24


class InvalidGeometry(ValueError):
    pass


class MassNormalizedELM:
    def __init__(self, m: float, config: Config | None = None):
        self.m = float(m)
        if self.m <= 1.0:
            raise ValueError("The pressure construction requires m > 1.")
        self.config = config or Config()
        self.k = self.m - 1.0
        self.nu = 1.0 / self.k
        self.Cm = self.m / self.k
        # These quantities are determined entirely by the prescribed u(x,0).
        self.s0 = np.sqrt(2.0 * self.m * (self.m + 1.0) / self.k)
        self.initial_shape_integral = np.exp(
            0.5 * np.log(np.pi)
            + gammaln(self.nu + 1.0)
            - gammaln(self.nu + 1.5)
        )
        self.M0 = self.s0 * self.initial_shape_integral
        self.Nv = self.config.pressure_neurons
        self.Ns = self.config.boundary_neurons
        self.nparams = self.Nv + 2 * self.Ns
        rng = np.random.default_rng(self.config.seed)
        self.W_v = rng.normal(size=(self.Nv, 2)) * self.config.pressure_frequency_scale
        self.b_v = rng.uniform(-np.pi, np.pi, self.Nv)
        # Independent hidden features and independent output coefficients.
        self.W_s = rng.normal(size=(2, self.Ns)) * self.config.boundary_frequency_scale
        self.b_s = rng.normal(size=(2, self.Ns))
        self.params = np.zeros(self.nparams)
        self.q_nodes, self.q_weights = roots_jacobi(
            self.config.mass_quad_n, self.nu, self.nu
        )
        self.history: list[dict] = []

    def boundary(self, t: np.ndarray, jacobian: bool = False) -> dict:
        t = np.atleast_1d(np.asarray(t, dtype=float))
        nt = t.size
        phase = t[:, None, None] * self.W_s[None, :, :] + self.b_s[None, :, :]
        # Initial-position anchoring uses only the given initial support.
        basis = np.sin(phase) - np.sin(self.b_s)[None, :, :]
        basis_t = np.cos(phase) * self.W_s[None, :, :]
        coeff = self.params[self.Nv :].reshape(2, self.Ns)
        positions = np.einsum("tsk,sk->ts", basis, coeff)
        positions += np.array([-self.s0, self.s0])[None, :]
        velocities = np.einsum("tsk,sk->ts", basis_t, coeff)
        c = positions.mean(axis=1)
        h = 0.5 * (positions[:, 1] - positions[:, 0])
        if np.any(h <= 0.0) or not np.all(np.isfinite(h)):
            raise InvalidGeometry("The two learned boundaries must satisfy s1 < s2.")
        result = dict(
            t=t, s=positions, sd=velocities, c=c, h=h,
            cd=velocities.mean(axis=1),
            hd=0.5 * (velocities[:, 1] - velocities[:, 0]),
        )
        if jacobian:
            js = np.zeros((nt, 2, self.nparams))
            jsd = np.zeros_like(js)
            js[:, 0, self.Nv : self.Nv + self.Ns] = basis[:, 0, :]
            js[:, 1, self.Nv + self.Ns :] = basis[:, 1, :]
            jsd[:, 0, self.Nv : self.Nv + self.Ns] = basis_t[:, 0, :]
            jsd[:, 1, self.Nv + self.Ns :] = basis_t[:, 1, :]
            result.update(
                js=js, jsd=jsd, jc=js.mean(axis=1),
                jh=0.5 * (js[:, 1] - js[:, 0]),
                jcd=jsd.mean(axis=1),
                jhd=0.5 * (jsd[:, 1] - jsd[:, 0]),
            )
        return result

    def _features(self, geom: dict, y: np.ndarray, jacobian: bool) -> dict:
        """Physical-coordinate F and total parameter derivatives at fixed y."""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = np.broadcast_to(y[None, :], (geom["t"].size, y.size))
        x = geom["c"][:, None] + geom["h"][:, None] * y
        xd = geom["cd"][:, None] + geom["hd"][:, None] * y
        spatial_phase = x[:, :, None] * self.W_v[None, None, :, 0]
        phase0 = geom["t"][:, None, None] * self.W_v[None, None, :, 1] + self.b_v[None, None, :]
        phase = spatial_phase + phase0
        sin, cos = np.sin(phase), np.cos(phase)
        # F(x,t)-F(0,t) is a pure gauge: normalization makes p unchanged.
        # Difference identities avoid subtracting two large common offsets.
        centered_sin = 2.0 * np.sin(0.5 * spatial_phase) * np.cos(phase0 + 0.5 * spatial_phase)
        centered_cos = -2.0 * np.sin(0.5 * spatial_phase) * np.sin(phase0 + 0.5 * spatial_phase)
        wx, wt = self.W_v[:, 0], self.W_v[:, 1]
        a = self.params[: self.Nv]
        fx_basis = cos * wx
        ft_basis = centered_cos * wt
        fxx_basis = -sin * wx**2
        f = centered_sin @ a
        fx, ft, fxx = fx_basis @ a, ft_basis @ a, fxx_basis @ a
        result = dict(y=y, x=x, xd=xd, f=f, fx=fx, ft=ft, fxx=fxx)
        if jacobian:
            fxxx = (-cos * wx**3) @ a
            fxt = (-sin * wx * wt) @ a
            jx = geom["jc"][:, None, :] + y[:, :, None] * geom["jh"][:, None, :]
            jxd = geom["jcd"][:, None, :] + y[:, :, None] * geom["jhd"][:, None, :]
            jf = fx[:, :, None] * jx
            jfx = fxx[:, :, None] * jx
            jft = fxt[:, :, None] * jx
            jfxx = fxxx[:, :, None] * jx
            jf[:, :, : self.Nv] += centered_sin
            jfx[:, :, : self.Nv] += fx_basis
            jft[:, :, : self.Nv] += ft_basis
            jfxx[:, :, : self.Nv] += fxx_basis
            result.update(jf=jf, jfx=jfx, jft=jft, jfxx=jfxx, jx=jx, jxd=jxd)
        return result

    def normalization(self, geom: dict, jacobian: bool = False) -> dict:
        """Log amplitude, its time derivative, and their exact parameter Jac.

        I = integral (1-y^2)^nu exp(nu F(c+h y,t)) dy.
        Gauss-Jacobi integrates the known endpoint weight; no pressure floor
        or additive epsilon is introduced into the conservation denominator.
        """
        f = self._features(geom, self.q_nodes, jacobian)
        logs = self.nu * f["f"]
        shift = logs.max(axis=1)
        weighted_exp = self.q_weights[None, :] * np.exp(logs - shift[:, None])
        sums = weighted_exp.sum(axis=1)
        omega = weighted_exp / sums[:, None]
        log_i = shift + np.log(sums)
        g = f["ft"] + f["fx"] * f["xd"]
        mean_g = np.sum(omega * g, axis=1)
        log_a = np.log(self.Cm) + self.k * (
            np.log(self.M0) - np.log(geom["h"]) - log_i
        )
        log_a_t = -self.k * geom["hd"] / geom["h"] - mean_g
        result = dict(log_a=log_a, log_a_t=log_a_t, log_i=log_i)
        if jacobian:
            mean_jf = np.einsum("tq,tqk->tk", omega, f["jf"])
            jg = (
                f["jft"] + f["jfx"] * f["xd"][:, :, None]
                + f["fx"][:, :, None] * f["jxd"]
            )
            mean_jg = np.einsum("tq,tqk->tk", omega, jg)
            covariance = np.einsum("tq,tq,tqk->tk", omega, g, f["jf"])
            covariance -= mean_g[:, None] * mean_jf
            jlog_a = -self.k * geom["jh"] / geom["h"][:, None] - mean_jf
            jlog_a_t = -self.k * (
                geom["jhd"] / geom["h"][:, None]
                - geom["hd"][:, None] * geom["jh"] / geom["h"][:, None]**2
            ) - mean_jg - self.nu * covariance
            result.update(jlog_a=jlog_a, jlog_a_t=jlog_a_t)
        return result

    def fields(self, geom: dict, norm: dict, y: np.ndarray, jacobian: bool = False) -> dict:
        """p, physical p_t/p_x/p_xx, and their total derivatives at fixed y."""
        f = self._features(geom, y, jacobian)
        y = f["y"]
        b, bp, bpp = 1.0 - y**2, -2.0 * y, -2.0
        h = geom["h"][:, None]
        velocity = f["xd"] / h
        log_e = norm["log_a"][:, None] + f["f"]
        with np.errstate(over="raise", invalid="raise"):
            e = np.exp(log_e)
            ct = b * (norm["log_a_t"][:, None] + f["ft"]) - bp * velocity
            cx = bp / h + b * f["fx"]
            cxx = (
                bpp / h**2 + 2.0 * bp * f["fx"] / h
                + b * (f["fxx"] + f["fx"]**2)
            )
            p, pt, px, pxx = e * b, e * ct, e * cx, e * cxx
        result = dict(p=p, pt=pt, px=px, pxx=pxx, y=y, x=f["x"])
        if jacobian:
            jlog_e = norm["jlog_a"][:, None, :] + f["jf"]
            jh = geom["jh"][:, None, :]
            jvelocity = f["jxd"] / h[:, :, None] - f["xd"][:, :, None] * jh / h[:, :, None]**2
            jct = b[:, :, None] * (norm["jlog_a_t"][:, None, :] + f["jft"])
            jct -= bp[:, :, None] * jvelocity
            jcx = -bp[:, :, None] * jh / h[:, :, None]**2 + b[:, :, None] * f["jfx"]
            jcxx = -2.0 * bpp * jh / h[:, :, None]**3
            jcxx = np.broadcast_to(jcxx, jlog_e.shape).copy()
            jcxx += 2.0 * bp[:, :, None] * (
                f["jfx"] / h[:, :, None]
                - f["fx"][:, :, None] * jh / h[:, :, None]**2
            )
            jcxx += b[:, :, None] * (
                f["jfxx"] + 2.0 * f["fx"][:, :, None] * f["jfx"]
            )
            result.update(
                jp=p[:, :, None] * jlog_e,
                jpt=e[:, :, None] * (ct[:, :, None] * jlog_e + jct),
                jpx=e[:, :, None] * (cx[:, :, None] * jlog_e + jcx),
                jpxx=e[:, :, None] * (cxx[:, :, None] * jlog_e + jcxx),
            )
        return result

    def assemble(self, t: np.ndarray, y: np.ndarray, jacobian: bool = True) -> tuple:
        geom = self.boundary(t, jacobian)
        norm = self.normalization(geom, jacobian)
        v = self.fields(geom, norm, y, jacobian)
        rpde = v["pt"] - self.k * v["p"] * v["pxx"] - v["px"]**2
        bc = self.fields(geom, norm, np.array([-1.0, 1.0]), jacobian)
        rd = geom["sd"] + bc["px"]
        g0 = self.boundary(np.array([0.0]), jacobian)
        n0 = self.normalization(g0, jacobian)
        y0, _ = np.polynomial.legendre.leggauss(self.config.initial_points)
        ic = self.fields(g0, n0, y0, jacobian)
        ric = ic["p"][0] - self.Cm * (1.0 - y0**2)
        wd, wi = np.sqrt(self.config.gamma_darcy), np.sqrt(self.config.gamma_initial)
        residual = np.concatenate([rpde.ravel(), wd * rd.ravel(), wi * ric])
        blocks = dict(
            pde=slice(0, rpde.size), darcy=slice(rpde.size, rpde.size + rd.size),
            initial=slice(rpde.size + rd.size, residual.size),
            pde_rms=float(np.sqrt(np.mean(rpde**2))),
            darcy_rms=float(np.sqrt(np.mean(rd**2))),
            initial_rms=float(np.sqrt(np.mean(ric**2))),
        )
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError("Non-finite pressure residual.")
        if not jacobian:
            return residual, None, blocks
        jpde = v["jpt"] - self.k * (
            v["p"][:, :, None] * v["jpxx"]
            + v["pxx"][:, :, None] * v["jp"]
        ) - 2.0 * v["px"][:, :, None] * v["jpx"]
        jd = geom["jsd"] + bc["jpx"]
        jac = np.concatenate([
            jpde.reshape(-1, self.nparams),
            wd * jd.reshape(-1, self.nparams),
            wi * ic["jp"][0],
        ], axis=0)
        if not np.all(np.isfinite(jac)):
            raise FloatingPointError("Non-finite analytic Jacobian.")
        return residual, jac, blocks

    def train(
        self, stages: list[float], *, svd_policy: str = "fixed",
        column_scaling: bool = True, svd_trunc_factor: float = 1.0e-6,
        iteration_callback=None, time_nodes=None,
    ) -> list[dict]:
        """Gauss-Newton, column scaling, and residual-decreasing backtracking.

        Fixed reference collocation points make assemble() a differentiable
        function of all pressure/boundary coefficients within each stage.
        No future pressure, density or boundary observations are supplied.

        residual_adaptive uses the existing project's Barenblatt rule:
        rcond = 1e-6 * ||R||_2 when ||R||_2 > 1, and 0 otherwise.
        The fixed-policy defaults reproduce the original implementation.
        column_scaling=False also matches the project's raw-J linear step.
        """
        if svd_policy not in ("fixed", "residual_adaptive", "residual_continuous"):
            raise ValueError("Unknown SVD policy.")
        if svd_trunc_factor <= 0 or not np.isfinite(svd_trunc_factor):
            raise ValueError("svd_trunc_factor must be finite and positive.")
        y, _ = np.polynomial.legendre.leggauss(self.config.space_points)
        for end_time in stages:
            if time_nodes is None:
                t = np.linspace(0.0, end_time, self.config.time_points)
            else:
                t = np.asarray(time_nodes, dtype=float)
                if (t.ndim != 1 or len(t)<2 or not np.isfinite(t).all()
                        or t[0]!=0. or t[-1]!=end_time or np.any(np.diff(t)<=0)):
                    raise ValueError('time_nodes must strictly increase from zero to the stage end.')
            check_t = np.linspace(0.0, end_time, max(257, 4 * len(t)))
            start_time = time.perf_counter()
            for iteration in range(self.config.max_iter):
                residual, jac, blocks = self.assemble(t, y, True)
                loss = float(np.mean(residual**2))
                residual_norm = float(np.linalg.norm(residual))
                rcond_used = self.config.rcond
                if svd_policy == "residual_adaptive":
                    rcond_used = svd_trunc_factor * residual_norm if residual_norm > 1.0 else 0.0
                elif svd_policy == "residual_continuous":
                    rcond_used = svd_trunc_factor * residual_norm
                row = dict(
                    m=self.m, stage=end_time, iteration=iteration, loss=loss,
                    pde_rms=blocks["pde_rms"], darcy_rms=blocks["darcy_rms"],
                    initial_rms=blocks["initial_rms"], step=0.0, step_rel=0.0,
                    rank=0, elapsed_seconds=time.perf_counter() - start_time,
                    residual_norm=residual_norm, rcond_used=rcond_used,
                    svd_policy=svd_policy, column_scaling=column_scaling,
                )
                if loss < self.config.residual_tol**2:
                    row["status"] = "residual_tolerance"
                    self.history.append(row)
                    if iteration_callback is not None:
                        iteration_callback(self, row)
                    print(f"m={self.m:g} T={end_time:g} iter={iteration} MSE={loss:.3e} converged", flush=True)
                    break
                column_norm = np.linalg.norm(jac, axis=0)
                active = column_norm > np.finfo(float).tiny
                column_scale = column_norm if column_scaling else np.ones(self.nparams)
                scaled_step, _, rank, _ = np.linalg.lstsq(
                    jac[:, active] / column_scale[active], -residual,
                    rcond=rcond_used,
                )
                delta = np.zeros(self.nparams)
                delta[active] = scaled_step / column_scale[active]
                row["rank"] = int(rank)
                row["delta_norm"] = float(np.linalg.norm(delta))
                old = self.params.copy()
                step = 1.0
                accepted = False
                for _ in range(self.config.max_backtracks):
                    self.params = old + step * delta
                    try:
                        self.boundary(check_t)
                        trial, _, _ = self.assemble(t, y, False)
                        trial_loss = float(np.mean(trial**2))
                    except (InvalidGeometry, FloatingPointError, OverflowError):
                        trial_loss = np.inf
                    if trial_loss < loss:
                        accepted = True
                        break
                    step *= 0.5
                if not accepted:
                    self.params = old
                    row["status"] = "line_search_stalled"
                    self.history.append(row)
                    if iteration_callback is not None:
                        iteration_callback(self, row)
                    print(f"m={self.m:g} T={end_time:g} iter={iteration} MSE={loss:.3e} line search stalled", flush=True)
                    break
                row["step"] = step
                row["step_rel"] = float(np.linalg.norm(step * delta) / (1.0 + np.linalg.norm(old)))
                row["status"] = "accepted"
                row["accepted_loss"] = trial_loss
                self.history.append(row)
                print(
                    f"m={self.m:g} T={end_time:g} iter={iteration:02d} "
                    f"MSE={loss:.3e}->{trial_loss:.3e} "
                    f"PDE={blocks['pde_rms']:.2e} Darcy={blocks['darcy_rms']:.2e} "
                    f"step={step:.2e} rank={rank}/{self.nparams} rcond={rcond_used:.2e}", flush=True,
                )
                if row["step_rel"] < self.config.step_tol:
                    row["status"] = "step_tolerance"
                    if iteration_callback is not None:
                        iteration_callback(self, row)
                    break
                if iteration_callback is not None:
                    iteration_callback(self, row)
        return self.history

    def predict(self, x: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the actual normalized field; never renormalize for export."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = np.broadcast_to(x[None, :], (t.size, x.size))
        geom = self.boundary(t)
        norm = self.normalization(geom)
        y = (x - geom["c"][:, None]) / geom["h"][:, None]
        spatial_phase = x[:, :, None] * self.W_v[None, None, :, 0]
        phase0 = t[:, None, None] * self.W_v[None, None, :, 1] + self.b_v[None, None, :]
        centered_sin = 2.0 * np.sin(0.5 * spatial_phase) * np.cos(phase0 + 0.5 * spatial_phase)
        f = centered_sin @ self.params[: self.Nv]
        b = np.maximum(1.0 - y**2, 0.0)
        p = np.zeros_like(b)
        inside = b > 0.0
        log_e = norm["log_a"][:, None] + f
        with np.errstate(over="raise", invalid="raise"):
            p[inside] = np.exp(log_e[inside]) * b[inside]
        u = (p / self.Cm)**self.nu
        return p, u

    def mass_curve(self, t: np.ndarray, n_quad: int | None = None, rule: str = "jacobi") -> np.ndarray:
        geom = self.boundary(t)
        norm = self.normalization(geom)
        n_quad = n_quad or self.config.mass_quad_n
        if rule == "jacobi":
            y, w = roots_jacobi(n_quad, self.nu, self.nu)
            f = self._features(geom, y, False)["f"]
            # The normalization still uses the original training quadrature.
            u_without_endpoint_weight = np.exp(self.nu * (norm["log_a"][:, None] + f - np.log(self.Cm)))
            return geom["h"] * (u_without_endpoint_weight @ w)
        if rule == "legendre":
            y, w = np.polynomial.legendre.leggauss(n_quad)
            p = self.fields(geom, norm, y, False)["p"]
            return geom["h"] * ((p / self.Cm)**self.nu @ w)
        raise ValueError("rule must be jacobi or legendre")

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, m=self.m, params=self.params, W_v=self.W_v, b_v=self.b_v,
            W_s=self.W_s, b_s=self.b_s, M0=self.M0, s0=self.s0,
            q_nodes=self.q_nodes, q_weights=self.q_weights,
            **{f"config_{key}": value for key, value in asdict(self.config).items()},
        )

    @classmethod
    def load(cls, path: str | Path) -> "MassNormalizedELM":
        with np.load(path) as data:
            if str(data.get('problem_family','')) == 'waiting_time_fractional_pressure_no_mass_pde':
                import sys
                waiting_dir = Path(__file__).resolve().parents[2]/'Waiting Time'/'mass_normalized_elm'
                if str(waiting_dir) not in sys.path:
                    sys.path.insert(0,str(waiting_dir))
                from fractional_no_mass_pde_solver import FractionalNoMassELM
                return FractionalNoMassELM.load(path)
            if str(data.get('problem_family','')) == 'waiting_time_squared_relu_no_mass':
                import sys
                waiting_dir = Path(__file__).resolve().parents[2]/'Waiting Time'/'mass_normalized_elm'
                if str(waiting_dir) not in sys.path:
                    sys.path.insert(0,str(waiting_dir))
                from relu_root_mass_solver import CompensatedReluNoMassWaitingELM
                return CompensatedReluNoMassWaitingELM.load(path)
            if str(data.get('problem_family','')) == 'waiting_time_standard_pressure_no_mass':
                import sys
                waiting_dir = Path(__file__).resolve().parents[2]/'Waiting Time'/'mass_normalized_elm'
                if str(waiting_dir) not in sys.path:
                    sys.path.insert(0,str(waiting_dir))
                from no_mass_waiting_solver import NoMassWaitingELM
                return NoMassWaitingELM.load(path)
            if str(data.get('problem_family','')) == 'waiting_time_sine_root_mass':
                import sys
                waiting_dir = Path(__file__).resolve().parents[2]/'Waiting Time'/'mass_normalized_elm'
                if str(waiting_dir) not in sys.path:
                    sys.path.insert(0,str(waiting_dir))
                from sine_root_mass_solver import SineRootMassWaitingELM
                return SineRootMassWaitingELM.load(path)
            if str(data.get('problem_family','')) == 'waiting_time_squared_relu_root_mass':
                import sys
                waiting_dir = Path(__file__).resolve().parents[2]/'Waiting Time'/'mass_normalized_elm'
                if str(waiting_dir) not in sys.path:
                    sys.path.insert(0,str(waiting_dir))
                from relu_root_mass_solver import ReluRootMassWaitingELM
                return ReluRootMassWaitingELM.load(path)
            if 'problem_family' in data and str(data['problem_family'])=='waiting_time_fixed_random_tau_jacobi':
                # These checkpoints live beside the experiment-specific head.
                import sys
                waiting_dir = Path(__file__).resolve().parents[2]/'Waiting Time'/'mass_normalized_elm'
                if str(waiting_dir) not in sys.path:
                    sys.path.insert(0,str(waiting_dir))
                if str(data.get('mass_rule',''))=='moving_positive_intervals_fractional_gauss_jacobi':
                    from root_mass_waiting_solver import RootMassWaitingELM
                    return RootMassWaitingELM.load(path)
                if str(data.get('stable_backend',''))=='mp_contracted_chebyshev_of_original_sines':
                    from chebyshev_waiting_solver import ChebyshevWaitingELM
                    return ChebyshevWaitingELM.load(path)
                from random_tau_jacobi_solver import RandomTauJacobiWaitingELM
                return RandomTauJacobiWaitingELM.load(path)
            if 'problem_family' in data and str(data['problem_family'])=='waiting_time_standard_pressure':
                from waiting_solver import WaitingTimeMassELM
                return WaitingTimeMassELM.load(path)
            if 'pressure_head' in data and str(data['pressure_head'])=='standard_pressure_elm':
                from standard_solver import StandardPressureMassELM
                return StandardPressureMassELM.load(path)
            if cls is MassNormalizedELM and "stable_backend" in data:
                from stable_solver import StableMassNormalizedELM
                return StableMassNormalizedELM.load(path)
            config = Config(**{
                key: data[f"config_{key}"].item() for key in asdict(Config())
            })
            model = cls(float(data["m"]), config)
            for name in ("params", "W_v", "b_v", "W_s", "b_s", "q_nodes", "q_weights"):
                setattr(model, name, data[name].copy())
        return model
