"""Stable arithmetic for the unchanged mass-normalized sine ELM.

Analytic residuals/Jacobians and np.linalg.lstsq are inherited unchanged.
Two independent boundary ELMs are evaluated in arbitrary precision. For
the pressure sine sum, arbitrary-precision contraction precedes conversion
to polynomial coefficients. Taylor truncation of the same sine features
is checked below 1e-30 for values and required derivatives on the domain.
The Gaussian normalization rule is the original saved rule, not changed.
No future solution or support information is available in this module.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import mpmath
import numpy as np

from solver import Config, InvalidGeometry, MassNormalizedELM


class StableMassNormalizedELM(MassNormalizedELM):
    def __init__(self, m: float, config: Config | None = None,
                 precision_digits: int = 50, minimum_degree: int = 112):
        super().__init__(m, config)
        if precision_digits<40 or minimum_degree<48:
            raise ValueError("Use at least 40 decimal digits and degree 48.")
        self.precision_digits = precision_digits
        self.minimum_degree = minimum_degree
        self.mp = mpmath.mp.clone()
        self.mp.dps = precision_digits
        self._prepared_signature = None
        self._phase_cache = {}
        self._coefficient_cache = {}
        self._boundary_basis_cache = {}
        self.maximum_degree_used = minimum_degree
        self.maximum_taylor_bound = 0.0

    def _mp_float(self, x):
        return self.mp.mpf(float(x))

    def _prepare(self) -> None:
        signature = tuple(a.tobytes() for a in (self.W_v, self.b_v, self.W_s, self.b_s))
        if signature==self._prepared_signature:
            return
        self._prepared_signature = signature
        self._wx = [self._mp_float(w) for w in self.W_v[:, 0]]
        self._wt = [self._mp_float(w) for w in self.W_v[:, 1]]
        self._b = [self._mp_float(b) for b in self.b_v]
        self._ws = [[self._mp_float(w) for w in row] for row in self.W_s]
        self._bs = [[self._mp_float(b) for b in row] for row in self.b_s]
        self._powers = [[self.mp.mpf(1) for _ in self._wx]]
        self._phase_cache.clear()
        self._coefficient_cache.clear()
        self._boundary_basis_cache.clear()

    @staticmethod
    def _cache_put(cache: dict, key, value, maximum: int = 4) -> None:
        cache[key] = value
        while len(cache)>maximum:
            del cache[next(iter(cache))]

    def boundary(self, t: np.ndarray, jacobian: bool = False) -> dict:
        self._prepare()
        t = np.atleast_1d(np.asarray(t, dtype=float))
        key = t.tobytes()
        if key not in self._boundary_basis_cache:
            values, derivatives = [], []
            for ti in t:
                value_sides, derivative_sides = [], []
                tt = self._mp_float(ti)
                for side in range(2):
                    value_sides.append([2*self.mp.sin(w*tt/2)*self.mp.cos(b+w*tt/2)
                                        for w,b in zip(self._ws[side],self._bs[side])])
                    derivative_sides.append([w*self.mp.cos(w*tt+b)
                                             for w,b in zip(self._ws[side],self._bs[side])])
                values.append(value_sides)
                derivatives.append(derivative_sides)
            self._cache_put(self._boundary_basis_cache, key, (values, derivatives))
        values, derivatives = self._boundary_basis_cache[key]
        coefficient = [[self._mp_float(a) for a in row]
                       for row in self.params[self.Nv:].reshape(2,self.Ns)]
        s = np.empty((t.size,2))
        sd = np.empty_like(s)
        c, h, cd, hd = (np.empty(t.size) for _ in range(4))
        for i in range(t.size):
            positions = [self.mp.fdot(values[i][side],coefficient[side])
                         +self._mp_float((-1 if side==0 else 1)*self.s0) for side in range(2)]
            velocities = [self.mp.fdot(derivatives[i][side],coefficient[side]) for side in range(2)]
            s[i] = [float(p) for p in positions]
            sd[i] = [float(v) for v in velocities]
            c[i],h[i] = float((positions[0]+positions[1])/2),float((positions[1]-positions[0])/2)
            cd[i],hd[i] = float((velocities[0]+velocities[1])/2),float((velocities[1]-velocities[0])/2)
        if np.any(h<=0) or not np.all(np.isfinite(h)):
            raise InvalidGeometry("The two learned boundaries must satisfy s1 < s2.")
        result = dict(t=t,s=s,sd=sd,c=c,h=h,cd=cd,hd=hd)
        if jacobian:
            js = np.zeros((t.size,2,self.nparams))
            jsd = np.zeros_like(js)
            for side in range(2):
                sl = slice(self.Nv+side*self.Ns,self.Nv+(side+1)*self.Ns)
                js[:,side,sl] = np.array([[float(v) for v in row[side]] for row in values])
                jsd[:,side,sl] = np.array([[float(v) for v in row[side]] for row in derivatives])
            result.update(js=js,jsd=jsd,jc=js.mean(axis=1),jh=.5*(js[:,1]-js[:,0]),
                          jcd=jsd.mean(axis=1),jhd=.5*(jsd[:,1]-jsd[:,0]))
        return result

    def _phases(self, t: np.ndarray) -> tuple:
        key = t.tobytes()
        if key not in self._phase_cache:
            phases, sine, cosine = [], [], []
            for ti in t:
                theta = [w*self._mp_float(ti)+b for w,b in zip(self._wt,self._b)]
                phases.append(theta)
                sine.append([self.mp.sin(z) for z in theta])
                cosine.append([self.mp.cos(z) for z in theta])
            self._cache_put(self._phase_cache,key,(phases,sine,cosine))
        return self._phase_cache[key]

    def _degree(self, radius: float) -> int:
        # Bound all required derivatives through order three, including Ft/Fxt.
        # Unit lower weights also bound individual pressure-parameter features.
        wx = np.abs(self.W_v[:,0])
        weights = np.maximum(np.abs(self.params[:self.Nv]),1.0)
        amplification = float(np.sum(weights*np.maximum(wx**3,1.0)*np.maximum(np.abs(self.W_v[:,1]),1.0)))
        z = max(float(wx.max())*radius,1.0)
        degree = self.minimum_degree
        while True:
            log_bound = math.log(amplification)+(degree-2)*math.log(z)-math.lgamma(degree-1)
            if log_bound<math.log(1e-30):
                break
            degree += 16
            if degree>512:
                raise FloatingPointError("The sine expansion requires more than degree 512.")
        self.maximum_degree_used = max(self.maximum_degree_used,degree)
        self.maximum_taylor_bound = max(self.maximum_taylor_bound,math.exp(log_bound))
        return degree

    def _coefficients(self, t: np.ndarray, radius: float) -> tuple:
        self._prepare()
        degree = self._degree(radius)
        while len(self._powers)<=degree:
            n = len(self._powers)
            self._powers.append([p*w/n for p,w in zip(self._powers[-1],self._wx)])
        key = (t.tobytes(),self.params[:self.Nv].tobytes(),degree)
        if key in self._coefficient_cache:
            return self._coefficient_cache[key]
        _,sines,cosines = self._phases(t)
        a = [self._mp_float(v) for v in self.params[:self.Nv]]
        coefficient = np.zeros((t.size,degree+1))
        coefficient_t = np.zeros_like(coefficient)
        for i in range(t.size):
            asin = [v*s for v,s in zip(a,sines[i])]
            acos = [v*c for v,c in zip(a,cosines[i])]
            asin_t = [v*w for v,w in zip(asin,self._wt)]
            acos_t = [v*w for v,w in zip(acos,self._wt)]
            for n in range(1,degree+1):
                sign = -1 if n%4 in (2,3) else 1
                value = asin if n%2==0 else acos
                value_t = acos_t if n%2==0 else asin_t
                sign_t = -1 if n%4 in (1,2) else 1
                coefficient[i,n] = float(sign*self.mp.fdot(value,self._powers[n]))
                coefficient_t[i,n] = float(sign_t*self.mp.fdot(value_t,self._powers[n]))
        result = (coefficient,coefficient_t)
        self._cache_put(self._coefficient_cache,key,result,maximum=8)
        return result

    @staticmethod
    def _polyval(coefficients: np.ndarray, x: np.ndarray, derivative: int = 0) -> np.ndarray:
        c = coefficients
        for _ in range(derivative):
            c = c[:,1:]*np.arange(1,c.shape[1])[None,:]
        result = np.zeros_like(x)
        for column in c[:,::-1].T:
            result = result*x+column[:,None]
        return result

    def shape_at(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t,dtype=float))
        x = np.asarray(x,dtype=float)
        if x.ndim==1:
            x = np.broadcast_to(x[None,:],(t.size,x.size))
        if x.size==0:
            return np.zeros_like(x)
        coefficient,_ = self._coefficients(t,float(np.max(np.abs(x))))
        return self._polyval(coefficient,x)

    def _features(self, geom: dict, y: np.ndarray, jacobian: bool) -> dict:
        y = np.asarray(y,dtype=float)
        if y.ndim==1:
            y = np.broadcast_to(y[None,:],(geom['t'].size,y.size))
        x = geom['c'][:,None]+geom['h'][:,None]*y
        xd = geom['cd'][:,None]+geom['hd'][:,None]*y
        radius = float(np.max(np.abs(geom['c'])+geom['h']))
        coefficient,coefficient_t = self._coefficients(geom['t'],radius)
        f = self._polyval(coefficient,x)
        fx = self._polyval(coefficient,x,1)
        ft = self._polyval(coefficient_t,x)
        fxx = self._polyval(coefficient,x,2)
        result = dict(y=y,x=x,xd=xd,f=f,fx=fx,ft=ft,fxx=fxx)
        if jacobian:
            # Individual Jacobian features carry no large output coefficient.
            # Time-phase sin/cos are computed before rounding in high precision.
            _,ss,cc = self._phases(geom['t'])
            s0 = np.array([[float(v) for v in row] for row in ss])[:,None,:]
            c0 = np.array([[float(v) for v in row] for row in cc])[:,None,:]
            spatial = x[:,:,None]*self.W_v[None,None,:,0]
            sine_x,cosine_x = np.sin(spatial),np.cos(spatial)
            cosine_minus_one = -2*np.sin(spatial/2)**2
            centered_sin = s0*cosine_minus_one+c0*sine_x
            centered_cos = c0*cosine_minus_one-s0*sine_x
            sine = s0*cosine_x+c0*sine_x
            cosine = c0*cosine_x-s0*sine_x
            wx,wt = self.W_v[:,0],self.W_v[:,1]
            fxxx = self._polyval(coefficient,x,3)
            fxt = self._polyval(coefficient_t,x,1)
            jx = geom['jc'][:,None,:]+y[:,:,None]*geom['jh'][:,None,:]
            jxd = geom['jcd'][:,None,:]+y[:,:,None]*geom['jhd'][:,None,:]
            jf = fx[:,:,None]*jx
            jfx = fxx[:,:,None]*jx
            jft = fxt[:,:,None]*jx
            jfxx = fxxx[:,:,None]*jx
            jf[:,:,:self.Nv] += centered_sin
            jfx[:,:,:self.Nv] += cosine*wx
            jft[:,:,:self.Nv] += centered_cos*wt
            jfxx[:,:,:self.Nv] += -sine*wx**2
            result.update(jf=jf,jfx=jfx,jft=jft,jfxx=jfxx,jx=jx,jxd=jxd)
        return result

    def predict(self, x: np.ndarray, t: np.ndarray) -> tuple:
        t = np.atleast_1d(np.asarray(t,dtype=float))
        x = np.asarray(x,dtype=float)
        if x.ndim==1:
            x = np.broadcast_to(x[None,:],(t.size,x.size))
        geom = self.boundary(t)
        norm = self.normalization(geom)
        b = np.maximum(1-((x-geom['c'][:,None])/geom['h'][:,None])**2,0)
        coefficient,_ = self._coefficients(t,float(np.max(np.abs(geom['c'])+geom['h'])))
        f = self._polyval(coefficient,x)
        p = np.zeros_like(b)
        inside = b>0
        with np.errstate(over='raise',invalid='raise'):
            log_e = norm['log_a'][:,None]+f
            p[inside] = np.exp(log_e[inside])*b[inside]
        return p,(p/self.Cm)**self.nu

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path,m=self.m,params=self.params,W_v=self.W_v,b_v=self.b_v,
            W_s=self.W_s,b_s=self.b_s,M0=self.M0,s0=self.s0,q_nodes=self.q_nodes,q_weights=self.q_weights,
            stable_backend='mp_contracted_sine_series',stable_precision_digits=self.precision_digits,
            stable_minimum_degree=self.minimum_degree,
            **{f'config_{key}':value for key,value in asdict(self.config).items()})

    @classmethod
    def load(cls, path: str | Path) -> 'StableMassNormalizedELM':
        with np.load(path) as data:
            config = Config(**{key:data[f'config_{key}'].item() for key in asdict(Config())})
            digits = int(data['stable_precision_digits']) if 'stable_precision_digits' in data else 50
            degree = int(data['stable_minimum_degree']) if 'stable_minimum_degree' in data else 112
            model = cls(float(data['m']),config,digits,degree)
            for name in ('params','W_v','b_v','W_s','b_s','q_nodes','q_weights'):
                setattr(model,name,data[name].copy())
        return model
