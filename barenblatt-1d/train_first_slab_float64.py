"""Float64 Cartesian sine evaluation for the time-slab solver."""
import numpy as np

from solver import InvalidGeometry
from time_slab_solver import TimeSlabPressureELM


class Float64TimeSlabELM(TimeSlabPressureELM):
    """Same standard sine ELM, evaluated directly in ordinary double precision."""
    def boundary(self, t, jacobian=False):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        half = .5*t[:, None, None]*self.W_s[None, :, :]
        values = 2*np.sin(half)*np.cos(self.b_s[None, :, :]+half)
        derivatives = self.W_s[None, :, :]*np.cos(
            t[:, None, None]*self.W_s[None, :, :]+self.b_s[None, :, :])
        coefficients = self.params[self.Nv:].reshape(2, self.Ns)
        s = self.initial_positions[None, :]+np.einsum('tsn,sn->ts', values, coefficients)
        sd = np.einsum('tsn,sn->ts', derivatives, coefficients)
        h, hd = .5*(s[:, 1]-s[:, 0]), .5*(sd[:, 1]-sd[:, 0])
        if np.any(h <= 0) or not np.all(np.isfinite(h)):
            raise InvalidGeometry('The two learned boundaries must satisfy s1 < s2.')
        result = dict(t=t, s=s, sd=sd, c=s.mean(axis=1), h=h,
                      cd=sd.mean(axis=1), hd=hd)
        if jacobian:
            js = np.zeros((t.size, 2, self.nparams))
            jsd = np.zeros_like(js)
            for side in range(2):
                sl = slice(self.Nv+side*self.Ns, self.Nv+(side+1)*self.Ns)
                js[:, side, sl], jsd[:, side, sl] = values[:, side], derivatives[:, side]
            result.update(js=js, jsd=jsd, jc=js.mean(axis=1),
                          jh=.5*(js[:, 1]-js[:, 0]), jcd=jsd.mean(axis=1),
                          jhd=.5*(jsd[:, 1]-jsd[:, 0]))
        return result

    def _sine(self, x, t):
        phase = x[:, :, None]*self.W_v[None, None, :, 0]
        phase += t[:, None, None]*self.W_v[None, None, :, 1]+self.b_v
        return np.sin(phase), np.cos(phase)

    def shape_at(self, x, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = np.broadcast_to(x[None, :], (t.size, x.size))
        return self._sine(x, t)[0]@self.params[:self.Nv]

    def _features(self, geom, y, jacobian):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = np.broadcast_to(y[None, :], (geom['t'].size, y.size))
        x = geom['c'][:, None]+geom['h'][:, None]*y
        xd = geom['cd'][:, None]+geom['hd'][:, None]*y
        sine, cosine = self._sine(x, geom['t'])
        wx, wt = self.W_v[:, 0], self.W_v[:, 1]
        a = self.params[:self.Nv]
        f, fx = sine@a, cosine@(wx*a)
        ft, fxx = cosine@(wt*a), sine@(-wx**2*a)
        result = dict(y=y, x=x, xd=xd, f=f, fx=fx, ft=ft, fxx=fxx)
        if jacobian:
            fxxx, fxt = cosine@(-wx**3*a), sine@(-wx*wt*a)
            jx = geom['jc'][:, None, :]+y[:, :, None]*geom['jh'][:, None, :]
            jxd = geom['jcd'][:, None, :]+y[:, :, None]*geom['jhd'][:, None, :]
            jf, jfx = fx[:, :, None]*jx, fxx[:, :, None]*jx
            jft, jfxx = fxt[:, :, None]*jx, fxxx[:, :, None]*jx
            jf[:, :, :self.Nv] += sine
            jfx[:, :, :self.Nv] += cosine*wx
            jft[:, :, :self.Nv] += cosine*wt
            jfxx[:, :, :self.Nv] -= sine*wx**2
            result.update(jf=jf, jfx=jfx, jft=jft, jfxx=jfxx, jx=jx, jxd=jxd)
        return result

    def state(self):
        state = super().state()
        state.update(evaluation_backend='numpy_float64_direct_sine',
                     model_loader='train_first_slab_float64.Float64TimeSlabELM.load')
        return state
