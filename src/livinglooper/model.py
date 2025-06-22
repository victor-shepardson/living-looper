import torch
import time
import math
from typing import Union

# wrap a predictor with mean subtraction
# maybe not so useful -- experience replay benefits from using its own 
# mean tracking, as the replayed data can be centered better
# class ZeroMean(torch.nn.Module):
#     """wrap another model to subtract the mean feature and target.
#     """
#     n:int
#     def __init__(self, m:torch.nn.Module):
#         super().__init__()
#         self.m = m
#         self.n_feat = m.n_feat
#         self.n_target = m.n_target
#         self.register_buffer('mu_x', torch.empty(self.n_feat))
#         self.register_buffer('mu_y', torch.empty(self.n_target))
#         self.n = 0

#     def reset(self):
#         self.m.reset()
#         self.mu_x.zero_()
#         self.mu_y.zero_()
#         self.n = 0

#     def finalize(self):
#         self.m.finalize()

#     def partial_fit(self, t:int, x, y):
#         n = self.n
#         self.mu_x[:] = self.mu_x * n/(n+1) + x/(n+1)
#         self.mu_y[:] = self.mu_y * n/(n+1) + y/(n+1)
#         self.n += 1
#         self.m.partial_fit(t, x-self.mu_x, y-self.mu_y)

#     def predict(self, t:int, x, temp:float=0.5):
#         y = self.m.predict(t, x-self.mu_x, temp=temp) + self.mu_y
#         return y

class Spherize(torch.nn.Module):
    def __init__(self, cls:type, n_feature:int, n_target:int):
        super().__init__()
        self.m = cls(n_feature, n_target+1)

    # wait this should be unconstrained, i.e. log?
    # currently it should be possible to get negative magnitudes
    def mag(self, y):
        return y.pow(2).sum().sqrt()[None] + 1e-7
        # return y.pow(2).sum().log().clip(-5, 5)[None]*2
    
    @torch.jit.export
    def reset(self):
        self.m.reset()

    @torch.jit.export
    def finalize(self):
        self.m.finalize()

    @torch.jit.export
    def partial_fit(self, t:int, x, y):
        y_ = torch.cat([self.mag(y), y])
        self.m.partial_fit(t, x, y_)

    @torch.jit.export
    def predict(self, t:int, x, temp:float=0.5):
        y_ = self.m.predict(t, x, temp=temp)
        mag, y = y_.split([1, y_.shape[-1]-1])
        y = y / self.mag(y) * mag
        # y = y * (mag - self.mag(y)).exp()
        return y

# NOTE: would be better to include a's prediction in the feature for b?
class Residual(torch.nn.Module):
    """wrap two models, with one predicting the residual of the other"""
    def __init__(self, a:torch.nn.Module, b:torch.nn.Module):
        super().__init__()
        self.a = a
        self.b = b
    
    @torch.jit.export
    def reset(self):
        self.a.reset()
        self.b.reset()

    @torch.jit.export
    def finalize(self):
        self.a.finalize()
        self.b.finalize()

    @torch.jit.export
    def partial_fit(self, t:int, x, y):
        self.a.finalize()
        y_ = self.a.predict(t, x)
        self.a.partial_fit(t, x, y)
        self.b.partial_fit(t, x, y-y_)

    @torch.jit.export
    def predict(self, t:int, x, temp:float=0.5):
        y = self.a.predict(t, x, temp=temp)
        y = y + self.b.predict(t, x, temp=temp)
        return y
    
class Residual2(torch.nn.Module):
    """wrap two models, with one predicting the residual of the other.
    this version provides the prediction of the first as an additional feature to the second.
    """
    def __init__(self, a:torch.nn.Module, cls:torch.nn.Module, n_feat:int, n_target:int, **kw):
        super().__init__()
        self.a = a
        self.b = cls(n_feat+n_target, n_target, **kw)
    
    @torch.jit.export
    def reset(self):
        self.a.reset()
        self.b.reset()

    @torch.jit.export
    def finalize(self):
        self.a.finalize()
        self.b.finalize()

    @torch.jit.export
    def partial_fit(self, t:int, x, y):
        self.a.finalize()
        y_ = self.a.predict(t, x)
        self.a.partial_fit(t, x, y)
        # print(y.shape, y_.shape)
        self.b.partial_fit(t, torch.cat((x, y_), -1), y-y_)

    @torch.jit.export
    def predict(self, t:int, x, temp:float=0.5):
        y_ = self.a.predict(t, x, temp=temp)
        y = y_ + self.b.predict(t, torch.cat((x, y_), -1), temp=temp)
        return y
    
class Residual3(torch.nn.Module):
    """wrap two models, with one predicting the residual of the other.
    this version provides the prediction of the first as the *only* feature to the second.
    """
    def __init__(self, a:torch.nn.Module, cls:torch.nn.Module, n_feat:int, n_target:int, **kw):
        super().__init__()
        self.a = a
        self.b = cls(n_target, n_target, **kw)
    
    @torch.jit.export
    def reset(self):
        self.a.reset()
        self.b.reset()

    @torch.jit.export
    def finalize(self):
        self.a.finalize()
        self.b.finalize()

    @torch.jit.export
    def partial_fit(self, t:int, x, y):
        # NOTE this may be expensive when the first model has a finalize step
        self.a.finalize()
        y_ = self.a.predict(t, x)
        self.a.partial_fit(t, x, y)
        # print(y.shape, y_.shape)
        self.b.partial_fit(t, y_, y-y_)

    @torch.jit.export
    def predict(self, t:int, x, temp:float=0.5):
        y_ = self.a.predict(t, x, temp=temp)
        y = y_ + self.b.predict(t, y_, temp=temp)
        return y

# idea: predict log(y**2) to enforce nonnegativity
class Moments(torch.nn.Module):
    """wrap another model to predict y, y**2, (y**3),
    and use these to estimate variance (skew) conditioned on input features.
    sample predictions from a (skew) normal distribution.
    """
    def __init__(self, cls:type, n_feat:int, n_target:int, n_moment=2, **kw):
        super().__init__()
        self.m = cls(n_feat, n_target*n_moment, **kw)
        self.n_moment = n_moment

    @torch.jit.export
    def reset(self):
        self.m.reset()

    @torch.jit.export
    def finalize(self):
        self.m.finalize()

    @torch.jit.export
    def partial_fit(self, t:int, x, y):
        y = torch.cat([y**m for m in range(1, self.n_moment+1)])
        self.m.partial_fit(t, x, y)

    @torch.jit.export
    def predict(self, t:int, x, temp:float=0.5):
        y = self.m.predict(t, x, temp=0.0)

        if self.n_moment==2:
            y, y2 = y.chunk(2, -1)
            if temp>0:
                # E[x^2] - E[x]^2
                w = (y2 - y**2).clip(0, 1).sqrt()
                y = y + w*temp*torch.randn_like(y)
        elif self.n_moment==3:
            y, y2, y3 = y.chunk(3, -1)
            if temp>0:
                # mean, standard deviation, skewness
                mu = y
                sigma = (y2 - y**2).clip(1e-3, 1).sqrt()
                gamma = ((y3 - 3*mu*sigma**2 - mu**3) / sigma**3).clip(-.995, .995)
                # skew normal parameters
                g23 = gamma.abs()**(2/3)
                delta2 = math.pi/2 * g23/(g23 + ((4-math.pi)/2)**(2/3))
                delta = gamma.sgn() * delta2.sqrt()
                alpha = delta / (1 - delta2).sqrt()
                omega = sigma / (1 - 2/math.pi*delta2).sqrt()
                xi = mu - omega*delta*(2/math.pi)**0.5
                # print('mu', mu)
                # print('sigma', sigma)
                # print('gamma', gamma)
                # print(xi, omega, alpha)
                y = randn_skew(xi, omega, alpha)

        return y

def randn_skew(loc, scale, alpha):
    #adapted from https://stackoverflow.com/questions/36200913/generate-n-random-numbers-from-a-skew-normal-distribution-using-numpy
    sigma = alpha / (1.0 + alpha**2)**0.5
    u0 = torch.randn_like(loc)
    v = torch.randn_like(loc)
    u1 = (sigma*u0 + (1.0 - sigma**2)**0.5*v) * scale
    u1[u0 < 0] *= -1
    u1 = u1 + loc
    return u1

# idea: online discretization + markov model
# part 1: online discretization
# part 2: variable order markov model
# part 3: nearest neighbor search for inference

# EM-like algorithm for online clustering, with different features and targets
class EM(torch.nn.Module):
    def __init__(self, n_feat:int, n_target:int, max_clusters:int=32):
        super().__init__()

        self.register_buffer('means', torch.zeros(max_clusters, n_feat))
        self.register_buffer('vars', torch.ones(max_clusters, n_feat))
        self.register_buffer('y_means', torch.zeros(max_clusters, n_target))
        self.register_buffer('y_vars', torch.ones(max_clusters, n_target))
        self.register_buffer('counts', torch.zeros(max_clusters))
        self.k = 0 # number of clusters
        self.max_clusters = max_clusters
        self.minvar = 0.1
        self.maxvar = 1
        self.reset()

    def reset(self):
        self.k = 0
        self.vars[:] = self.minvar
        self.y_vars[:] = self.minvar

    @torch.jit.export
    def finalize(self):
        pass

    @torch.jit.export
    def partial_fit(self, t:int, x, y):
        """
        Args:
            t: unused
            x: Tensor[n_feat]
            y: Tensor[n_target]
        """
        # online discretization
        # probabilistic; EM without reassignment
        #   maintain mean and variance for each cluster
        #   assign to highest likelihood, 
        #     considering candidate new cluster (with prior variance)
        #    -> this would always make a second cluster from the second point
        #     candidate variance has to be higher than actual new cluster,
        #       otherwise new cluster would always be better
        #       it would work to update new cluster variance to 1/2, i.e.
        #       mean of prior 1 and sample 0
        # ^ unused (e.g., inactive loop) features are a problem here
        #   if all 0s for example their variance will just shrink to the minimum
        #   causing one cluster as runaway most plausible...

        # put mean into kth slot
        if self.k < self.max_clusters:
            self.means[self.k] = x
            k_slice = self.k + 1
        else:
            k_slice = self.k
        # compute likelihoods
        lik = self.loglik(k_slice, x)
        # select max index
        i = lik.argmax()
        # update means, counts, vars, possibly k
        if i>=self.k:
            self.k += 1
        # print('t', t, 'lik\n', lik, 'k', self.k, 'i', i)
        # print('lik\n', lik)
        print('i', i.item())
        self.counts[i] += 1
        n = self.counts[i]
        # first update should set mean to data value
        self.means[i] = (self.means[i] * (n-1) + x)/n
        # first update should retain some of prior -- since measured var is 0
        self.vars[i] = ((
            self.vars[i] * n + (x-self.means[i])**2
            )/(1+n)).clip(self.minvar, self.maxvar)
        # self.vars[i] = self.minvar

        self.y_means[i] = (self.y_means[i] * (n-1) + y)/n
        self.y_vars[i] = ((
            self.y_vars[i] * n + (y-self.y_means[i])**2
            )/(1+n)).clip(self.minvar, self.maxvar)
        
        # print('fit y', y[:3], self.y_means[i,:3], self.y_vars[i,:3])

    @torch.jit.export
    def predict(self, t:int, x, temp:float=0.5):#, itemp:float=0.0):
        if self.k==0:
            return self.y_means[0]
        
        itemp = temp
        
        lik = self.loglik(self.k, x)
        lik = lik - lik.max()
        # print(lik)
        lik = lik.exp()
        # print(lik)
        if itemp==0:
            i = lik.argmax()
        else:
            i = torch.multinomial(lik.pow(1/itemp), 1).squeeze()
        y = self.y_means[i] 
        y = y + temp*torch.randn_like(y)*self.y_vars[i].sqrt()
        print('i', i)
        # print('i', i, 'y', y[:3])
        return y

    def loglik(self, k:int, x):
        mu = self.means[:k]
        var = self.vars[:k]
        # return ((-(mu-x)**2 / (2*var)).exp() * var.pow(-0.5)).sum(1)
        return (-(mu-x)**2 / (2*var) - 0.5 * var.log()).sum(1)



class ILR(torch.nn.Module):
    """
    Incremental Linear Regression.

    Complexity is N**2 in the feature size
    """
    n:int
    def __init__(self, n_feat:int, n_target:int):
        super().__init__()
        self.n = 0
        self.n_feat = n_feat
        self.n_target = n_target
        self.register_buffer('P', torch.empty(n_feat, n_feat))
        self.register_buffer('w', torch.empty(n_feat, n_target))
        self.register_buffer('mu_x', torch.empty(n_feat))
        self.register_buffer('mu_y', torch.empty(n_target))

        self.reset()

    def reset(self):
        self.n = 0
        self.w.zero_()
        self.P[:] = torch.eye(self.n_feat)*100
        self.mu_x.zero_()
        self.mu_y.zero_()

    @torch.jit.export
    def finalize(self):
        # print(self.mu_y)
        pass

    @torch.jit.export
    def partial_fit(self, t:int, x, y):
        """
        Args:
            t: unused
            x: Tensor[n_feat]
            y: Tensor[n_target]
        """
        n = self.n
        self.mu_x[:] = self.mu_x * n/(n+1) + x/(n+1)
        self.mu_y[:] = self.mu_y * n/(n+1) + y/(n+1)
        self.n += 1
        # print('y', y[0], 'mu', self.mu_y[0], 'n', self.n)

        x = x - self.mu_x
        y = y - self.mu_y

        x = x[:,None] # [n_feat, 1]
        y = y[None] # [1, n_target]

        Px = self.P@x # n_feat, 1
        k = Px / (1 + x.T@Px) # n_feat, 1
        self.P.sub_(k @ Px.T) # n_feat, n_feat
        self.w.add_(k * (y - x.T@self.w))
        #                       \1, n_target/
        #              \  n_feat, n_target /

    @torch.jit.export
    def predict(self, t:int, x, temp:float=0.0):
        # return self.mu_y
        return (x - self.mu_x) @ self.w + self.mu_y


class IPLS(torch.nn.Module):
    """
    Incremental Partial Least Squares

    Following the NI-PALS algorithm as described by Abdi 
    (https://onlinelibrary.wiley.com/doi/abs/10.1002/wics.51), 
    and extending the CIPLS algorithm from Jordao et al 
    (https://ieeexplore.ieee.org/document/9423374/)
    for multivariate targets.
    improved numerical stability and performance by using an inner loop, 
    burn-in steps, and a prior on the score norm. 

    Complexity is n_latent * (n_target + n_feat)
    (plus the cost of pinv for [n_target x n_latent])
    """
    def __init__(self,
        n_feat:int,
        n_target:int,
        n_latent:int = 8,
        burn_in:int = 3,
        inner_steps:int = 2,
        ):
        super().__init__()
        self.n_feat = n_feat
        self.n_target = n_target
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

        n = self.n
        self.mu_x[:] = self.mu_x * n/(n+1) + x/(n+1)
        self.mu_y[:] = self.mu_y * n/(n+1) + y/(n+1)
        self.n[:] = self.n + 1

        x = x - self.mu_x
        y = y - self.mu_y

        # burn-in steps: update only means, W, t norm
        # TODO: does this harm estimation of b?
        if self.n <= self.burn_in:
            # tss = self.t_sq_sum

            self.Wz[:] = self.Wz + x*self.u[:,None]
            tz = (self.Wz @ x) / (self.Wz.pow(2).sum(1).sqrt()+1e-7)
            self.t_sq_sum[:] = self.t_sq_sum + tz*tz
        else:
            # loop over target dimensions
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
                # print('i', i)
                # print('x', x.abs().mean())
                # print('t', t)
                # print('P', self.P.abs().mean())

                x = x - t*self.P[i]
                y = y - b*t*C_i

    
    @torch.jit.export
    def finalize(self):
        if self.n > self.burn_in:
            C = self.Cz / torch.linalg.vector_norm(
                self.Cz, 2, 1, keepdim=True)
            b = self.bz / self.t_sq_sum.sqrt()
            # print(self.P)
            # self.H[:] = torch.linalg.pinv(self.P) * b @ C
            S = torch.linalg.lstsq(self.P, b[:,None]*C, driver='gelsd')
            # print(S)
            self.H[:] = S.solution
        # print(self.mu_y.chunk(3, -1))

    @torch.jit.export
    def predict(self, t:int, x, temp:float=0.0):
        # return self.mu_y
        y = (x - self.mu_x)@self.H + self.mu_y
        return y


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
            n_feat:int, n_target:int, memory_size:int=4096, n_func:int=32):
        super().__init__()
        self.n_func = n_func
        self.mem_size = memory_size
        self.lr_start = 1.0
        self.lr_decay = 0.977
        self.weight_decay = 1e-2
        # self.amp_decay = 1e-1
        self.amp_decay = 1e-2
        # self.amp_decay = 1e-3
        self.n = 32#64
        self.batch_size = 16
        self.momentum = 0.85
        self.data_size = 0
        self.lr = self.lr_start
        # freq_per_func = n_target
        freq_per_func = 1
        # parameters: phase, frequency, amplitude of N periodic functions
        self.register_buffer('memory', torch.empty(memory_size, n_target))
        self.register_buffer('t_memory', torch.empty(memory_size, dtype=torch.long))

        # self.register_buffer('amp', torch.empty(self.n_func, n_target, dtype=torch.complex64))
        self.register_buffer('amp', torch.empty(self.n_func, n_target))
        self.register_buffer('phase', torch.empty(self.n_func, n_target))

        self.register_buffer('freq', torch.empty(self.n_func, freq_per_func))
        self.register_buffer('center_freq', 
            torch.linspace(0.01, 0.99, self.n_func)[:,None].logit())
        self.register_buffer('bias', torch.empty(n_target))

        # damn, complex autograd doesn't work with JIT
        # it was silently failing since amp is initialized to all real,
        # apparently (?)
        # self.register_buffer('amp_grad', torch.empty(self.n_func, n_target, dtype=torch.complex64))
        self.register_buffer('amp_grad', torch.empty(self.n_func, n_target))
        self.register_buffer('phase_grad', torch.empty(self.n_func, n_target))
        self.register_buffer('freq_grad', torch.empty(self.n_func, freq_per_func))

        self.reset()

    def reset(self):
        self.data_size = 0
        self.lr = self.lr_start
        self.amp_grad.zero_()
        self.freq_grad.zero_()
        self.phase_grad.zero_()
        self.amp.fill_(1/self.n_func**0.5)
        self.phase[:] = torch.rand_like(self.phase)
        # self.freq[:] = torch.linspace(0.01, 0.99, self.n_func)[:,None].logit()
        self.freq.zero_()
        self.bias.zero_()

    def get_amp(self, amp):
        # return (amp.exp() + 1).log()
        # return amp.abs()
        return amp
        # return torch.nn.functional.relu(amp)

    def get_freq(self, freq):
        freq = self.center_freq + freq.tanh()*64/self.n_func
        return freq.sigmoid()/2
    
    @torch.jit.export
    def evaluate(self, t):
        return self._evaluate(t, self.amp, self.freq, self.bias)

    def _evaluate(self, t, amp, freq, phase):
        # t: Tensor[batch]
        t = t[:,None,None]
        # Tensor[batch, basis, target]
        a = self.get_amp(amp)
        fr = self.get_freq(freq)
        z = (((t * fr + phase) * 2 * math.pi).sin() * a).sum(1) / self.n_func**0.5
        # z = ((t * fr * 2j * math.pi).exp() * a).real.sum(1) / self.n_func**0.5
        # Tensor[batch, target]
        z = z + self.bias
        return z

    def sample_wedge(self, n:int):
        r = torch.rand(2,n)
        return (r.sum(0) - 1).abs()

    @torch.jit.export
    def predict(self, t:int, x, temp:float=0.0):
        z = self.evaluate(torch.full((1,), float(t)))[0]
        return z

    @torch.jit.export
    def finalize(self):
        pass
        # print(self.get_freq(self.freq))

    @torch.jit.export
    def partial_fit(self, t:int, x, z):

        write_idx = self.data_size % self.mem_size
        # print(write_idx, self.mem_size)
        self.memory[write_idx] = z
        self.t_memory[write_idx] = t
        self.data_size += 1

        self.lr *= self.lr_decay
        # print(self.lr)

        # compute target / feature mean exactly
        n = self.data_size
        self.bias[:] = self.bias * n/(n+1) + z/(n+1)

        # TODO: crazy idea: resample everything to noninteger t for experience replay?
        i = self.data_size - 1
        ts = i - (self.sample_wedge(self.n) * min(i, self.mem_size)).long()
        ts = ts % self.mem_size

        # print(samps)
        batches = ts.chunk(self.n//self.batch_size)
        for batch in batches:
            amp = self.amp.clone().requires_grad_()
            freq = self.freq.clone().requires_grad_()
            phase = self.phase.clone().requires_grad_()
            self.amp_grad.mul_(self.momentum)
            self.freq_grad.mul_(self.momentum)
            self.phase_grad.mul_(self.momentum)

            t_batch, z_batch = (
                self.t_memory[batch], self.memory[batch])
            z_ = self._evaluate(t_batch, amp, freq, phase)
            # print(t.shape, x.shape, x_.shape)

            # amp penalty for sparsity
            loss = self.get_amp(amp).abs().sqrt().sum() * self.amp_decay

            # loss with just periodic functions
            loss = loss + (z_batch - z_).abs().sum(-1).mean()

            # TODO: harmonicity loss?
            # print(f0.shape, self.freq.shape)

            ag, fg, pg = torch.autograd.grad(
                [loss], [amp, freq, phase],
                )

            if torch.jit.isinstance(ag, torch.Tensor):
                ag = ag / (torch.linalg.vector_norm(ag.flatten()) + 1)
                self.amp_grad.add_(ag)
                self.amp.sub_(self.amp_grad*self.lr)
            if torch.jit.isinstance(fg, torch.Tensor):
                fg = fg / (torch.linalg.vector_norm(fg.flatten()) + 1)
                self.freq_grad.add_(fg)
                self.freq.sub_(self.freq_grad*self.lr)
            if torch.jit.isinstance(pg, torch.Tensor):
                pg = pg / (torch.linalg.vector_norm(pg.flatten()) + 1)
                self.phase_grad.add_(pg)
                self.phase.sub_(self.phase_grad*self.lr)
                self.phase.remainder_(1.0)


# class GDKRR(torch.nn.Module):
#     """Gradient Descent Kernel Regression with Residual prediction"""
#     n_func:int
#     lr_start:float
#     lr_decay:float
#     lr:float
#     amp_decay:float
#     momentum:float
#     n:int
#     batch_size:int
#     data_size:int
#     def __init__(self, 
#             n_feat:int, n_target:int, memory_size:int=4096, n_func:int=32):
#         super().__init__()
#         self.n_func = n_func
#         self.mem_size = memory_size
#         self.lr_start = 1.0
#         self.lr_decay = 0.977
#         self.weight_decay = 1e-2
#         # self.amp_decay = 1e-1
#         self.amp_decay = 1e-2
#         # self.amp_decay = 1e-3
#         self.mm_lr = 3e-2
#         self.n_hidden = 128
#         self.n = 32#64
#         self.batch_size = 16
#         self.momentum = 0.85
#         self.data_size = 0
#         self.lr = self.lr_start
#         # freq_per_func = n_target
#         freq_per_func = 1
#         # parameters: phase, frequency, amplitude of N periodic functions
#         self.register_buffer('memory', torch.empty(memory_size, n_target))
#         self.register_buffer('t_memory', torch.empty(memory_size, dtype=torch.long))
#         self.register_buffer('feat_memory', torch.empty(memory_size, n_feat))
#         self.register_buffer('amp', torch.empty(self.n_func, n_target, dtype=torch.complex64))
#         # self.register_buffer('freq', torch.empty(self.n_func, 1))
#         self.register_buffer('freq', torch.empty(self.n_func, freq_per_func))
#         self.register_buffer('center_freq', 
#             torch.linspace(0.01, 0.99, self.n_func)[:,None].logit())
#         self.register_buffer('ih_proj', torch.empty(n_feat+n_target, self.n_hidden))
#         self.register_buffer('h_bias', torch.empty(self.n_hidden))
#         self.register_buffer('hr_proj', torch.empty(self.n_hidden, n_target))
#         self.register_buffer('feat_mean', torch.empty(n_feat))
#         self.register_buffer('bias', torch.empty(n_target))

#         self.register_buffer('amp_grad', torch.empty(self.n_func, n_target, dtype=torch.complex64))
#         self.register_buffer('freq_grad', torch.empty(self.n_func, freq_per_func))
#         self.register_buffer('ih_proj_grad', torch.empty_like(self.ih_proj))
#         self.register_buffer('hr_proj_grad', torch.empty_like(self.hr_proj))
#         self.register_buffer('h_bias_grad', torch.empty_like(self.h_bias))

#         self.reset()

#     def reset(self):
#         self.data_size = 0
#         self.lr = self.lr_start
#         self.amp_grad.zero_()
#         self.freq_grad.zero_()
#         self.ih_proj[:] = torch.randn_like(self.ih_proj).mul(self.ih_proj.shape[0]**-0.5)
#         self.hr_proj[:] = torch.randn_like(self.hr_proj).mul(self.hr_proj.shape[0]**-0.5 * 1e-1)
#         self.h_bias.zero_()
#         self.amp.fill_(1/self.n_func**0.5)
#         # self.freq[:] = torch.linspace(0.01, 0.99, self.n_func)[:,None].logit()
#         self.freq.zero_()
#         self.bias.zero_()
#         self.feat_mean.zero_()
#         self.ih_proj_grad.zero_()
#         self.hr_proj_grad.zero_()
#         self.h_bias_grad.zero_()

#     def get_amp(self, amp):
#         # return (amp.exp() + 1).log()
#         # return amp.abs()
#         return amp
#         # return torch.nn.functional.relu(amp)

#     def get_freq(self, freq):
#         freq = self.center_freq + freq.tanh()*64/self.n_func
#         return freq.sigmoid()/2
    
#     def get_resid(self, x, z, ih_proj, h_bias, hr_proj):
#         h = torch.cat((x-self.feat_mean, z-self.bias), -1) @ ih_proj + h_bias
#         h = h.tanh()
#         r = h @ hr_proj
#         # r = r / (r.abs() + 1)
#         return r

#     @torch.jit.export
#     def evaluate(self, t):
#         return self._evaluate(t, self.amp, self.freq, self.bias)

#     def _evaluate(self, t, amp, freq, bias):
#         # t: Tensor[batch]
#         t = t[:,None,None]
#         # Tensor[batch, basis, target]
#         a = self.get_amp(amp)
#         fr = self.get_freq(freq)
#         # z = (((t * fr + p) * 2 * math.pi).sin() * a).sum(1) / self.n_func**0.5
#         z = ((t * fr * 2j * math.pi).exp() * a).real.sum(1) / self.n_func**0.5
#         # Tensor[batch, target]
#         z = z + bias
#         return z

#     def sample_wedge(self, n:int):
#         r = torch.rand(2,n)
#         return (r.sum(0) - 1).abs()

#     @torch.jit.export
#     def predict(self, t:Union[torch.Tensor, int], x, temp:float=0.0):
#         z = self.evaluate(torch.full((1,), float(t)))[0]
#         z = z + self.get_resid(x, z, self.ih_proj, self.h_bias, self.hr_proj)
#         return z

#     @torch.jit.export
#     def finalize(self):
#         pass
#         # print(self.get_freq(self.freq))

#     @torch.jit.export
#     def partial_fit(self, t:int, x, z):

#         write_idx = self.data_size % self.mem_size
#         # print(write_idx, self.mem_size)
#         self.memory[write_idx] = z
#         self.feat_memory[write_idx] = x
#         self.t_memory[write_idx] = t
#         self.data_size += 1

#         self.lr *= self.lr_decay
#         # print(self.lr)

#         # compute target / feature mean exactly
#         n = self.data_size
#         self.bias[:] = self.bias * n/(n+1) + z/(n+1)
#         self.feat_mean[:] = self.feat_mean * n/(n+1) + x/(n+1)

#         # TODO: crazy idea: resample everything to noninteger t for experience replay?
#         i = self.data_size - 1
#         ts = i - (self.sample_wedge(self.n) * min(i, self.mem_size)).long()
#         ts = ts % self.mem_size

#         # print(samps)
#         batches = ts.chunk(self.n//self.batch_size)
#         for batch in batches:
#             amp = self.amp.clone().requires_grad_()
#             freq = self.freq.clone().requires_grad_()
#             ih_proj = self.ih_proj.clone().requires_grad_()
#             h_bias = self.h_bias.clone().requires_grad_()
#             hr_proj = self.hr_proj.clone().requires_grad_()
#             self.amp_grad.mul_(self.momentum)
#             self.freq_grad.mul_(self.momentum)
#             self.ih_proj_grad.mul_(self.momentum)
#             self.h_bias_grad.mul_(self.momentum)
#             self.hr_proj_grad.mul_(self.momentum)

#             t_batch, z_batch, x_batch = (
#                 self.t_memory[batch], self.memory[batch], self.feat_memory[batch])
#             z_ = self._evaluate(t_batch, amp, freq, self.bias)
#             # print(t.shape, x.shape, x_.shape)

#             # amp penalty for sparsity
#             loss = self.get_amp(amp).abs().sqrt().sum() * self.amp_decay

#             # loss with just periodic functions
#             loss = loss + (z_batch - z_).abs().sum(-1).mean()

#             # TODO: harmonicity loss?
#             # print(f0.shape, self.freq.shape)

#             # residual estimation from feature
#             # resid_ = self.get_resid(x_batch, z_.detach(), res_proj)
#             resid_ = self.get_resid(x_batch, z_, ih_proj, h_bias, hr_proj)
#             # # print(resid_.abs().max())
#             z_ = z_+resid_
#             # z_ = z_.detach() + resid_
#             res_loss = (z_batch - z_).abs().sum(-1).mean()
#             # # weight decay
#             res_loss = res_loss + (
#                 self.ih_proj.abs().sum()
#                 + self.h_bias.abs().sum()
#                 + self.hr_proj.abs().sum()
#              ) * self.weight_decay
            
#             loss = loss + res_loss * self.mm_lr

#             ag, fg, ihg, hbg, hrg = torch.autograd.grad(
#                 [loss], [amp, freq, ih_proj, h_bias, hr_proj],
#                 )

#             if torch.jit.isinstance(ag, torch.Tensor):
#                 ag = ag / (torch.linalg.vector_norm(ag.flatten()) + 1)
#                 self.amp_grad.add_(ag)
#                 self.amp.sub_(self.amp_grad*self.lr)
#                 print(self.amp)
#             if torch.jit.isinstance(fg, torch.Tensor):
#                 fg = fg / (torch.linalg.vector_norm(fg.flatten()) + 1)
#                 self.freq_grad.add_(fg)
#                 self.freq.sub_(self.freq_grad*self.lr)
#             if torch.jit.isinstance(ihg, torch.Tensor):
#                 # print(ihg.abs().max())
#                 ihg = ihg / (torch.linalg.vector_norm(ihg.flatten()) + 1)
#                 self.ih_proj_grad.add_(ihg)
#                 self.ih_proj.sub_(self.ih_proj_grad*self.lr)
#             if torch.jit.isinstance(hbg, torch.Tensor):
#                 # print(hbg.abs().max())
#                 hbg = hbg / (torch.linalg.vector_norm(hbg.flatten()) + 1)
#                 self.h_bias_grad.add_(hbg)
#                 self.h_bias.sub_(self.h_bias_grad*self.lr)
#             if torch.jit.isinstance(hrg, torch.Tensor):
#                 # print(hrg.abs().max())
#                 hrg = hrg / (torch.linalg.vector_norm(hrg.flatten()) + 1)
#                 self.hr_proj_grad.add_(hrg)
#                 self.hr_proj.sub_(self.hr_proj_grad*self.lr)


def fit_model(X, Y, model_cls, **kw):
    """create a model, fit to data, print timing and return the predictor"""
    T = torch.arange(X.shape[0])
    n_feat = X.shape[-1]
    n_target = Y.shape[-1]

    model = model_cls(n_feat, n_target, **kw)

    # model.partial_fit(0, torch.randn(n_feat), torch.randn(n_target))
    # model.reset()
    model = torch.jit.script(model)
    # torch.jit.save(model, 'test.ts')
    # model = torch.jit.load('test.ts')

    ts = []

    for t,x,y in zip(T,X,Y):
        t_ns = time.time_ns()
        # print(t, x, y)
        # print(model.w)
        model.partial_fit(t,x,y)
        ts.append(time.time_ns() - t_ns)
        # break

    t_ns = time.time_ns()
    model.finalize()

    ts = torch.tensor(ts).float()#[1:]

    print(f'partial_fit time (mean): {torch.mean(ts)*1e-6} ms')
    print(f'partial_fit time (99%): {torch.quantile(ts, 0.99)*1e-6} ms')
    print(f'finalize time: {(time.time_ns() - t_ns)*1e-6} ms')
    
    return model