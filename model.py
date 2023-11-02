import torch
import time

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
    def partial_fit(self, x, y):
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
    def predict(self, x):
        return (x - self.mu_x)@self.H + self.mu_y


def fit(X, Y, n_latent=8, burn_in=2, inner_steps=2):
    """create an IPLS model, fit to data, print timing and return the predictor"""
    n_feat = X.shape[-1]
    n_target = Y.shape[-1]

    ipls = IPLS(n_feat, n_target, n_latent,
        burn_in=burn_in, inner_steps=inner_steps)
    
    ipls = torch.jit.script(ipls)
    
    ts = []

    for x,y in zip(X,Y):
        t_ns = time.time_ns()
        ipls.partial_fit(x,y)
        ts.append(time.time_ns() - t_ns)

    t_ns = time.time_ns()
    ipls.finalize()

    print(f'partial_fit time (mean): {torch.mean(ts)*1e-6} ms')
    print(f'partial_fit time (99%): {torch.quantile(ts, 0.99)*1e-6} ms')
    print(f'finalize time: {(time.time_ns() - t_ns)*1e-6} ms')
    
    return ipls.predict