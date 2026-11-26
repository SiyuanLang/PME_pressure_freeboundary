"""Standard sine-pressure ELM with the user's mass rescaling only.

p_raw = sum a_j sin(wx_j*x + wt_j*t + b_j)
M_raw = integral ((m-1)/m * maximum(p_raw, 0))**(1/(m-1)) dx
p = (M0/M_raw)**(m-1) * p_raw.

No envelope, exponential pressure head, pressure clipping, or pressure floor.
Positive-part density is the explicitly selected convention. Analytic Jac
is classical on each fixed Gaussian active set; a zero at a Gaussian node
has no classical derivative and is rejected during residual evaluation.
Boundary, stable arithmetic and np.linalg.lstsq Gauss--Newton are inherited.
Only the prescribed initial datum is used for initialization and training.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.special import roots_jacobi, roots_legendre

from solver import Config
from stable_solver import StableMassNormalizedELM


class StandardPressureMassELM(StableMassNormalizedELM):
    pressure_head = 'standard_pressure_elm'

    def __init__(self, m: float, config: Config | None = None,
                 precision_digits: int = 50, minimum_degree: int = 112):
        super().__init__(m, config, precision_digits, minimum_degree)
        # Weight compensation is quadrature only, not a pressure envelope.
        self.q_weights = self.q_weights/(1-self.q_nodes**2)**self.nu

    def _coefficients(self, t: np.ndarray, radius: float) -> tuple:
        coefficient, coefficient_t = super()._coefficients(t, radius)
        _, sines, cosines = self._phases(t)
        a = [self._mp_float(v) for v in self.params[:self.Nv]]
        at = [v*w for v,w in zip(a,self._wt)]
        for i in range(t.size):
            # Restoring n=0 gives the full standard ELM, without centering.
            coefficient[i,0] = float(self.mp.fdot(a,sines[i]))
            coefficient_t[i,0] = float(self.mp.fdot(at,cosines[i]))
        return coefficient, coefficient_t

    def _features(self, geom: dict, y: np.ndarray, jacobian: bool) -> dict:
        f = super()._features(geom,y,jacobian)
        if jacobian:
            _,ss,cc = self._phases(geom['t'])
            s0 = np.array([[float(v) for v in row] for row in ss])[:,None,:]
            c0 = np.array([[float(v) for v in row] for row in cc])[:,None,:]
            spatial = f['x'][:,:,None]*self.W_v[None,None,:,0]
            sine = s0*np.cos(spatial)+c0*np.sin(spatial)
            cosine = c0*np.cos(spatial)-s0*np.sin(spatial)
            f['jf'][:,:,:self.Nv] = sine
            f['jft'][:,:,:self.Nv] = cosine*self.W_v[:,1]
        return f

    def initialize_pressure(self, end_time: float) -> dict:
        """Fit a stationary extension of the given initial datum, for a guess.

        These initialization targets do not become residual rows. They are
        neither future exact observations nor a prescribed future trajectory.
        Zero coefficients cannot initialize a mass-rescaled pressure network.
        """
        t = np.linspace(0,end_time,self.config.time_points)
        y,_ = roots_legendre(self.config.initial_points)
        x = self.s0*y
        phase = (x[None,:,None]*self.W_v[None,None,:,0]
                 +t[:,None,None]*self.W_v[None,None,:,1]+self.b_v)
        basis = np.sin(phase).reshape(-1,self.Nv)
        target = np.broadcast_to(self.Cm*(1-y**2),(t.size,y.size)).ravel()
        a,_,rank,_ = np.linalg.lstsq(basis,target,rcond=1e-12)
        self.params[:self.Nv] = a
        self.params[self.Nv:] = 0.0
        geom = self.boundary(t)
        norm = self.normalization(geom)
        return dict(initialization='least_squares_stationary_extension_of_given_initial_pressure',
                    initial_fit_rank=int(rank),initial_fit_rcond=1e-12,
                    initial_fit_rms=float(np.sqrt(np.mean((basis@a-target)**2))),
                    initial_raw_mass_min=float(np.exp(norm['log_m_raw']).min()),
                    future_exact_data_used=False,extra_training_residuals=False)

    def normalization(self, geom: dict, jacobian: bool = False) -> dict:
        f = self._features(geom,self.q_nodes,jacobian)
        q = f['f']
        if np.any(q==0):
            raise FloatingPointError('Positive-part density has no classical derivative at a Gaussian node with zero pressure.')
        active = q>0
        qp = np.maximum(q,0)
        z = qp**self.nu @ self.q_weights
        if np.any(z<=0) or not np.all(np.isfinite(z)):
            raise FloatingPointError('The unnormalized density must have finite positive mass.')
        q1 = np.zeros_like(q)
        q1[active] = qp[active]**(self.nu-1)
        g = f['ft']+f['fx']*f['xd']
        s = np.sum(self.q_weights*q1*g,axis=1)
        log_m = np.log(geom['h'])-self.nu*np.log(self.Cm)+np.log(z)
        log_a = self.k*(np.log(self.M0)-log_m)
        log_a_t = -self.k*geom['hd']/geom['h']-s/z
        result = dict(log_a=log_a,log_a_t=log_a_t,log_i=np.log(z),log_m_raw=log_m,
                      minimum_raw_pressure_at_mass_nodes=float(q.min()),
                      negative_mass_node_count=int((~active).sum()))
        if jacobian:
            js = np.einsum('tq,tqk->tk',self.q_weights*q1,f['jf'])
            jg = f['jft']+f['jfx']*f['xd'][:,:,None]+f['fx'][:,:,None]*f['jxd']
            d_s = np.einsum('tq,tqk->tk',self.q_weights*q1,jg)
            if self.nu!=1:
                q2 = np.zeros_like(q)
                q2[active] = qp[active]**(self.nu-2)
                d_s += (self.nu-1)*np.einsum('tq,tqk->tk',self.q_weights*q2*g,f['jf'])
            jlog_a = -self.k*geom['jh']/geom['h'][:,None]-js/z[:,None]
            jlog_a_t = -self.k*(geom['jhd']/geom['h'][:,None]
                -geom['hd'][:,None]*geom['jh']/geom['h'][:,None]**2)
            jlog_a_t -= d_s/z[:,None]
            jlog_a_t += self.nu*s[:,None]*js/z[:,None]**2
            result.update(jlog_a=jlog_a,jlog_a_t=jlog_a_t)
        return result

    def fields(self, geom: dict, norm: dict, y: np.ndarray, jacobian: bool = False) -> dict:
        f = self._features(geom,y,jacobian)
        a = np.exp(norm['log_a'])[:,None]
        at = norm['log_a_t'][:,None]
        p = a*f['f']
        pt = a*(at*f['f']+f['ft'])
        px,pxx = a*f['fx'],a*f['fxx']
        result = dict(p=p,pt=pt,px=px,pxx=pxx,y=f['y'],x=f['x'])
        if jacobian:
            jl = norm['jlog_a'][:,None,:]
            result.update(jp=a[:,:,None]*(f['jf']+f['f'][:,:,None]*jl),
                jpx=a[:,:,None]*(f['jfx']+f['fx'][:,:,None]*jl),
                jpxx=a[:,:,None]*(f['jfxx']+f['fxx'][:,:,None]*jl),
                jpt=pt[:,:,None]*jl+a[:,:,None]*(
                    norm['jlog_a_t'][:,None,:]*f['f'][:,:,None]
                    +at[:,:,None]*f['jf']+f['jft']))
        return result

    def _initial_pressure_target(self, y: np.ndarray, geom: dict) -> np.ndarray:
        """Prescribed initial datum; overridden by other benchmark problems."""
        return self.Cm*(1-y**2)

    def assemble(self, t: np.ndarray, y: np.ndarray, jacobian: bool = True) -> tuple:
        geom = self.boundary(t,jacobian)
        norm = self.normalization(geom,jacobian)
        v = self.fields(geom,norm,y,jacobian)
        rpde = v['pt']-self.k*v['p']*v['pxx']-v['px']**2
        bc = self.fields(geom,norm,np.array([-1.0,1.0]),jacobian)
        rzero = bc['p']
        rd = geom['sd']+bc['px']
        g0 = self.boundary(np.array([0.0]),jacobian)
        n0 = self.normalization(g0,jacobian)
        y0,_ = roots_legendre(self.config.initial_points)
        ic = self.fields(g0,n0,y0,jacobian)
        ric = ic['p'][0]-self._initial_pressure_target(y0,g0)
        wb,wi = np.sqrt(self.config.gamma_darcy),np.sqrt(self.config.gamma_initial)
        residual = np.concatenate([rpde.ravel(),wb*rzero.ravel(),wb*rd.ravel(),wi*ric])
        i0,i1,i2 = rpde.size,rpde.size+rzero.size,rpde.size+rzero.size+rd.size
        blocks = dict(pde=slice(0,i0),dirichlet=slice(i0,i1),darcy=slice(i1,i2),initial=slice(i2,residual.size),
                      pde_rms=float(np.sqrt(np.mean(rpde**2))),
                      dirichlet_rms=float(np.sqrt(np.mean(rzero**2))),
                      darcy_rms=float(np.sqrt(np.mean(rd**2))),initial_rms=float(np.sqrt(np.mean(ric**2))))
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError('Non-finite standard-pressure residual.')
        if not jacobian:
            return residual,None,blocks
        jpde = v['jpt']-self.k*(v['p'][:,:,None]*v['jpxx']+v['pxx'][:,:,None]*v['jp'])
        jpde -= 2*v['px'][:,:,None]*v['jpx']
        jd = geom['jsd']+bc['jpx']
        jac = np.concatenate([jpde.reshape(-1,self.nparams),wb*bc['jp'].reshape(-1,self.nparams),
                              wb*jd.reshape(-1,self.nparams),wi*ic['jp'][0]],axis=0)
        if not np.all(np.isfinite(jac)):
            raise FloatingPointError('Non-finite standard-pressure analytic Jacobian.')
        return residual,jac,blocks

    def predict(self, x: np.ndarray, t: np.ndarray) -> tuple:
        t = np.atleast_1d(np.asarray(t,dtype=float))
        x = np.asarray(x,dtype=float)
        if x.ndim==1:
            x = np.broadcast_to(x[None,:],(t.size,x.size))
        geom = self.boundary(t)
        norm = self.normalization(geom)
        inside = (x>=geom['s'][:,0,None]) & (x<=geom['s'][:,1,None])
        q = self.shape_at(np.where(inside,x,0),t)
        p = np.where(inside,np.exp(norm['log_a'])[:,None]*q,0)
        return p,(np.maximum(p,0)/self.Cm)**self.nu

    def mass_curve(self, t: np.ndarray, n_quad: int | None = None, rule: str = 'jacobi') -> np.ndarray:
        geom = self.boundary(t)
        norm = self.normalization(geom)
        if n_quad is None:
            nodes,weights = self.q_nodes,self.q_weights
        elif rule=='jacobi':
            nodes,weights = roots_jacobi(n_quad,self.nu,self.nu)
            weights = weights/(1-nodes**2)**self.nu
        elif rule=='legendre':
            nodes,weights = roots_legendre(n_quad)
        else:
            raise ValueError('Unknown Gaussian mass integration rule.')
        q = self._features(geom,nodes,False)['f']
        return geom['h']*np.exp(self.nu*norm['log_a'])*(np.maximum(q,0)**self.nu @ weights)/self.Cm**self.nu

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path,m=self.m,params=self.params,W_v=self.W_v,b_v=self.b_v,
            W_s=self.W_s,b_s=self.b_s,M0=self.M0,s0=self.s0,q_nodes=self.q_nodes,q_weights=self.q_weights,
            pressure_head=self.pressure_head,density_convention='positive_part',
            stable_backend='mp_contracted_standard_sine_series',stable_precision_digits=self.precision_digits,
            stable_minimum_degree=self.minimum_degree,
            **{f'config_{key}':value for key,value in asdict(self.config).items()})

    @classmethod
    def load(cls, path: str | Path) -> 'StandardPressureMassELM':
        with np.load(path,allow_pickle=False) as data:
            config = Config(**{key:data[f'config_{key}'].item() for key in asdict(Config())})
            model = cls(float(data['m']),config,int(data['stable_precision_digits']),int(data['stable_minimum_degree']))
            for name in ('params','W_v','b_v','W_s','b_s','q_nodes','q_weights'):
                setattr(model,name,data[name].copy())
        return model
