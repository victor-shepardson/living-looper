from typing import Dict

import torch

from model import IPLS, GDKR
from representation import Window, RNN
from transform import LinQuad, Tanh, Id

class FeatureStore(torch.nn.Module):
    max_steps: int
    memory: Dict[int, torch.Tensor]
    def __init__(self, 
            max_steps:int,
            ):
        super().__init__()
        self.max_steps = max_steps
        self.memory = {}

    def reset(self):
        # print('store reset')
        self.memory.clear()

    def add(self, step:int, feat):
        for k in list(self.memory):
            if step - k > self.max_steps:
                self.memory.pop(k)
        self.memory[step] = feat
        # print('memory contains:', self.memory.keys())

    def get(self, step:int):
        if step in self.memory:
            return self.memory[step]
        else:
            print('memory contains:', self.memory.keys())
            raise KeyError(step)
        

class Loop(torch.nn.Module):
    n_loops:int
    index:int
    verbose:int

    def __init__(self, 
            index:int,
            n_loops:int,
            n_context:int, # time dimension of model feature
            n_latent:int,
            latency_correct:int,
            limit_margin:torch.Tensor,
            verbose:int=0
        ):
        super().__init__()

        self.verbose = verbose

        self.index = index
        self.n_loops = n_loops
        self.n_latent = n_latent

        # self.rep = RNN(n_latent, 1024, 256)
        self.rep = Window(n_latent, n_context)

        # this assumes all loops use the same (sized) feature --
        # would need multiple stages of init to allow otherwise
        n_feature = n_loops * self.rep.n_feature #n_loops * n_context * n_latent

        self.store = FeatureStore(2*latency_correct)

        # for now assuming xform preserves target size
        self.target_xform = LinQuad()
        # self.target_xform = Id()

        # self.feat_xform = Tanh()
        self.feat_xform = Id()

        # self.model = GDKR(n_feature, n_latent)
        n_latent_ipls = 16
        # n_latent_ipls = n_latent
        self.model = IPLS(
            n_feat=n_feature, n_target=n_latent, n_latent=n_latent_ipls)

        self.register_buffer('limit_margin', limit_margin)
        self.register_buffer('z_min', 
            torch.empty(n_latent, requires_grad=False))
        self.register_buffer('z_max', 
            torch.empty(n_latent, requires_grad=False))

    def reset(self):
        self.model.reset()
        # self.store.reset() # this only gets reset by LivingLooper.reset
        self.rep.reset() # should this get reset?
        self.z_min.fill_(torch.inf)
        self.z_max.fill_(-torch.inf)

    def partial_fit(self, t:int, x, z):
        """fit raw feature x to raw target z"""
        self.z_min[:] = torch.minimum(self.z_min, z)
        self.z_max[:] = torch.maximum(self.z_max, z)
        # print('min', self.z_min, 'max', self.z_max)
        x = self.feat_xform(x)
        z = self.target_xform(z)
        self.model.partial_fit(t, x, z)

    def predict(self, t:int, x):
        """predict from raw feature x"""
        x = self.feat_xform(x)
        z = self.model.predict(t, x)
        z = self.target_xform.inv(z)
        z = z.clamp(self.z_min-self.limit_margin, self.z_max+self.limit_margin)
        z[~z.isfinite()] = 0
        if not z.isfinite().all():
            print('WARNING: nonfinite z')
        # print(self.index+1, z)
        return z

    def finalize(self):
        self.model.finalize()

    def feed(self, step:int, z):
        """
        use the value `z`
        to store the feature for time `step`
        """
        offset = 0
        while (step+offset) in self.store.memory:
            if self.verbose > 1:
                print('WARNING: feature already computed for loop', self.index+1, 'step', step+offset)
            offset += 1

        if self.verbose > 1:
            print('feed loop', self.index+1, 'offset', offset, 'step', step)
        # TODO: could add logic to allow features which don't support rollback here
        # i.e. if feed returns None, don't add?
        self.store.add(step, self.rep.feed(z, offset))

    def get(self, step:int):
        """get the feature for time `step`"""
        if self.verbose > 1:
            print(f'get loop', self.index+1, 'step', step)
        return self.store.get(step)