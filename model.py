import torch
import time
import math

class IPLS(torch.nn.Module):
    """
    Incremental Partial Least Squares

    Following the NIPALS algorithm as described by Abdi 
    (https://onlinelibrary.wiley.com/doi/abs/10.1002/wics.51), 
    and extending the CIPLS algorithm from Jordao et al 
    (https://ieeexplore.ieee.org/document/9423374/)
    for multivariate targets.
    improved numerical stability and performance by using an inner loop, 
    burn-in steps, and a prior on the data means and score norm. 
    """
    def __init__(self,
        n_feat:int,
        n_target:int,
        n_latent:int = 8,
        burn_in:int = 3,
        inner_steps:int = 2,
        ):
        super().__init__()
        self.burn_in = burn_in
        self.inner_steps = inner_steps

        self.register_buffer('mu_x', torch.empty(n_feat))
        self.register_buffer('mu_y', torch.empty(n_target))

        # target scores
        self.register_buffer('u', torch.empty(n_latent))

        # unnormalized W (feature weights)
        self.register_buffer('Wz', torch.empty(n_latent, n_feat)) 
        # unnormalized C (target weights)
        self.register_buffer('Cz', torch.empty(n_latent, n_target)) 
        # squared norm of tz (unnormalized feature scores)
        self.register_buffer('t_sq_sum', torch.empty(n_latent)) 
        # unnormalized b (regression weights)
        self.register_buffer('bz', torch.empty(n_latent))
        # feature loadings
        self.register_buffer('P', torch.empty(n_latent, n_feat))
        # steps count
        self.register_buffer('n', torch.empty((1,), dtype=torch.long))

        # prediction weights
        self.register_buffer('H', torch.empty(n_feat, n_target))

        self.reset()

    @torch.jit.export
    def reset(self):
        self.mu_x.zero_()
        self.mu_y.zero_()

        u0 = torch.randn_like(self.u)
        self.u[:] = u0 / u0.pow(2).sum().sqrt()

        self.Wz.zero_()
        self.Cz.zero_()
        #
        self.t_sq_sum[:] = 1
        self.bz.zero_()
        self.P.zero_()

        self.H.zero_()

        self.n.zero_()

    @torch.jit.export
    def partial_fit(self, _:int, x, y):
        self.n[:] = self.n + 1
        # prior expectation the means are 0
        self.mu_x[:] = (self.mu_x * self.n + x) / (self.n+1)
        self.mu_y[:] = (self.mu_y * self.n + y) / (self.n+1)

        x = x - self.mu_x
        y = y - self.mu_y

        # burn-in steps: update only means, W, t norm
        # TODO: does this harm estimation of b?
        if self.n <= self.burn_in:
            tss = self.t_sq_sum

            self.Wz[:] = self.Wz + x*self.u[:,None]
            tz = (self.Wz @ x) / (self.Wz.pow(2).sum(1).sqrt()+1e-7)
            self.t_sq_sum[:] = self.t_sq_sum + tz*tz
        else:
            for i in range(self.u.shape[0]):
                tss = self.t_sq_sum[i]

                # the NIPALS inner loop modifies u until t converges
                # CIPLS is simplified by having univariate targets, and has no u 
                # assumption: with temporal iteration u,t will converge well enough
                # but this can be improved with a fixed number of inner loop steps

                wz = self.Wz[i] + x*self.u[i]
                tz = x.dot(wz) / (wz.pow(2).sum().sqrt()+1e-7)
                tss_ = tss + tz*tz
                
                t = tz / tss_.sqrt()
                cz = self.Cz[i] + y*t
                C_i = cz / cz.pow(2).sum().sqrt()
                self.u[i] = y.dot(C_i)

                for _ in range(self.inner_steps-1):
                    wz = self.Wz[i] + x*self.u[i]
                    tz = x.dot(wz) / (wz.pow(2).sum().sqrt()+1e-7)
                    tss_ = tss + tz*tz
                    
                    t = tz / tss_.sqrt()
                    cz = self.Cz[i] + y*t
                    C_i = cz / cz.pow(2).sum().sqrt()
                    self.u[i] = y.dot(C_i)

                self.Wz[i] = wz
                self.t_sq_sum[i] = tss_

                self.Cz[i] = cz

                self.bz[i] = self.bz[i] + self.u[i]*tz
                b = self.bz[i] / self.t_sq_sum[i].sqrt()

                self.P[i] = self.P[i] + x*t

                x = x - t*self.P[i]
                y = y - b*t*C_i

    
    @torch.jit.export
    def finalize(self):
        C = self.Cz / torch.linalg.vector_norm(
            self.Cz, 2, 1, keepdim=True)
        b = self.bz / self.t_sq_sum.sqrt()
        # print(self.P)
        self.H[:] = torch.linalg.pinv(self.P) * b @ C

    @torch.jit.export
    def predict(self, t:int, x):
        return (x - self.mu_x)@self.H + self.mu_y


class GDKR(torch.nn.Module):
    """Gradient Descent Kernel Regression"""
    n_func:int
    lr_start:float
    lr_decay:float
    lr:float
    amp_decay:float
    momentum:float
    n:int
    batch_size:int
    data_size:int
    def __init__(self, 
            n_feat:int, n_target:int, memory_size:int=4096, n_func:int=128):
        super().__init__()
        self.n_func = n_func
        self.lr_start = 1.0
        self.lr_decay = 0.977
        self.amp_decay = 1e-2
        # self.amp_decay = 1e-3
        self.n = 64
        self.batch_size = 16
        self.momentum = 0.85
        self.data_size = 0
        self.lr = self.lr_start
        # parameters: phase, frequency, amplitude of N periodic functions
        self.register_buffer('memory', torch.empty(memory_size, n_target))
        self.register_buffer('t_memory', torch.empty(memory_size, dtype=torch.long))
        self.register_buffer('amp', torch.empty(self.n_func, n_target))
        # self.register_buffer('freq', torch.empty(self.n_func, 1))
        self.register_buffer('freq', torch.empty(self.n_func, n_target))
        self.register_buffer('center_freq', 
            torch.linspace(0.01, 0.99, self.n_func)[:,None].logit())
        self.register_buffer('phase', torch.empty(self.n_func, n_target))
        self.register_buffer('bias', torch.empty(n_target))
        self.register_buffer('amp_grad', torch.empty(self.n_func, n_target))
        self.register_buffer('freq_grad', torch.empty(self.n_func, n_target))
        self.register_buffer('phase_grad', torch.empty(self.n_func, n_target))
        # self.register_buffer('bias_grad', torch.empty(n_target))
        self.reset()

    def reset(self):
        self.data_size = 0
        self.lr = self.lr_start
        self.amp_grad.zero_()
        self.freq_grad.zero_()
        self.phase_grad.zero_()
        # self.bias_grad.zero_()
        self.amp.fill_(1/self.n_func**0.5)
        # self.freq[:] = torch.linspace(0.01, 0.99, self.n_func)[:,None].logit()
        self.freq.zero_()
        self.phase[:] = torch.rand_like(self.phase)
        self.bias.zero_()

    def get_amp(self, amp):
        # return (amp.exp() + 1).log()
        # return amp.abs()
        return amp
        # return torch.nn.functional.relu(amp)

    def get_freq(self, freq):
        freq = self.center_freq + freq.tanh()*64/self.n_func
        return freq.sigmoid()/2

    def get_phase(self, phase):
        return phase

    @torch.jit.export
    def evaluate(self, t):
        return self._evaluate(t, self.amp, self.freq, self.phase, self.bias)

    def _evaluate(self, t, amp, freq, phase, bias):
        # t: Tensor[batch]
        t = t[:,None,None]
        # Tensor[batch, basis, target]
        a = self.get_amp(amp)
        fr = self.get_freq(freq)
        p = self.get_phase(phase)
        x = (((t * fr + p) * 2 * math.pi).sin() * a).sum(1) / self.n_func**0.5
        # Tensor[batch, target]
        return x + bias

    def sample_wedge(self, n:int):
        r = torch.rand(2,n)
        return (r.sum(0) - 1).abs()

    def predict(self, t:int, x):
        return self.evaluate(torch.full((1,), float(t)))[0]

    def finalize(self):
        pass
        # print(self.get_freq(self.freq))

    @torch.jit.export
    def partial_fit(self, t:int, x, z):
    #     with torch.inference_mode(False):
    #         self._partial_fit(t, x, z)

    # def _partial_fit(self, t:int, x, z):
        self.memory[self.data_size] = z
        self.t_memory[self.data_size] = t
        self.data_size += 1
        # self.lr.mul_(self.lr_decay)
        self.lr *= self.lr_decay
        # print(self.lr)

        self.bias[:] = (self.bias * self.data_size + z) / (self.data_size + 1)

        i = self.data_size - 1
        ts = i - (self.sample_wedge(self.n) * i).long()

        # print(ts)

        assert (ts < self.data_size).all()
        assert (ts>=0).all()
        # TODO: deal with finite memory

        # print(samps)
        batches = ts.chunk(self.n//self.batch_size)
        for batch in batches:
            amp = self.amp.clone().requires_grad_()
            freq = self.freq.clone().requires_grad_()
            phase = self.phase.clone().requires_grad_()
            bias = self.bias#.clone().requires_grad_()
            self.amp_grad.mul_(self.momentum)
            self.freq_grad.mul_(self.momentum)
            self.phase_grad.mul_(self.momentum)
            # self.bias_grad.mul_(self.momentum)

            t_batch, z_batch = self.t_memory[batch], self.memory[batch]
            z_ = self._evaluate(t_batch, amp, freq, phase, bias)
            # print(t.shape, x.shape, x_.shape)

            loss = (z_batch - z_).abs().sum(-1).mean()
            # amp penalty for sparsity
            loss = loss + self.get_amp(amp).abs().sum() * self.amp_decay
            # TODO: harmonicity loss?
            # print(f0.shape, self.freq.shape)

            ag, pg, fg = torch.autograd.grad(
                [loss], [amp, phase, freq],
                )
            # ag, pg, fg, bg = torch.autograd.grad(
            #     [loss], [amp, phase, freq, bias],
            #     )
            if torch.jit.isinstance(ag, torch.Tensor):
                self.amp_grad.add_(ag)
                self.amp.sub_(self.amp_grad*self.lr)
            if torch.jit.isinstance(fg, torch.Tensor):
                self.freq_grad.add_(fg)
                # self.freq.sub_(self.freq_grad*self.lr / 3)
                self.freq.sub_(self.freq_grad*self.lr)
                # print(fg.shape, fg)
            # else:
                # print(fg)
            if torch.jit.isinstance(pg, torch.Tensor):
                self.phase_grad.add_(pg)
                self.phase.sub_(self.phase_grad*self.lr / 1)
                self.phase.remainder_(1)
            # if torch.jit.isinstance(bg, torch.Tensor):
            #     self.bias_grad.add_(bg)
            #     self.bias.sub_(self.bias_grad*self.lr)
        

def fit_model(X, Y, cls, **kw):
    """create a model, fit to data, print timing and return the predictor"""
    T = torch.arange(X.shape[0])
    n_feat = X.shape[-1]
    n_target = Y.shape[-1]

    model = cls(n_feat, n_target, **kw)
    
    model = torch.jit.script(model)
    
    ts = []

    for t,x,y in zip(T,X,Y):
        t_ns = time.time_ns()
        model.partial_fit(t,x,y)
        ts.append(time.time_ns() - t_ns)

    t_ns = time.time_ns()
    model.finalize()

    print(f'partial_fit time (mean): {torch.mean(ts)*1e-6} ms')
    print(f'partial_fit time (99%): {torch.quantile(ts, 0.99)*1e-6} ms')
    print(f'finalize time: {(time.time_ns() - t_ns)*1e-6} ms')
    
    return model