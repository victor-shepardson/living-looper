from typing import Dict, List, Optional, Union
import logging

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np

from fire import Fire
from termcolor import colored
from tqdm import tqdm

import nn_tilde

from .loop import Loop


# each loop has:
    # rep: (represents history as a feature vector) 
    #   feed: target -> ()  
    #   get: () -> feature
    # xform: (invertible, maps targets to/from prediction space)
    #   fwd: target -> target'
    #   inv: target' -> target
    # model: (supervised learner for autoregressive generation)
    #   partial_fit: [features], target -> ()
    #   finalize: () -> ()
    #   predict: [features] -> target

# a rep can be composed from other reps
# e.g. Tanh . Window

# LivingLooper
    # z = model.encode(x)
    # model.finalize on loops switching active -> inactive
    # (model.reset on loops switching inactive -> active)
    # feat_self = [l.rep.get for l in loops]
    # feat_other = delay(feat_self)
    # eval inactive loops:
    #    feat = combine(feat_self, feat_other, l)
    #    z[l] = l.model.predict(feat)
    # fit active loops:
    #    l.model.partial_fit(feat, z[l])
    # for all loops:
    #    l.rep.feed(z[l])

class LivingLooper(nn_tilde.Module):
    # __constants__ = ['loops']

    sampling_rate:int
    block_size:int

    prev_loop_index:int
    step:int
    record_length:int

    verbose:int

    encode_temp:bool

    def __init__(self, 
            model:torch.jit.ScriptModule, 
            n_loops:int, 
            n_context:int, # time dimension of model feature
            sr:int, # sample rate
            limit_margin:List[float], # limit latents relative to recorded min/max
            latent_signs:List[int], # flip latents
            verbose:int=0
            ):
        super().__init__()

        self.register_attribute('loop_index', (0,))
        self.register_attribute('thru', (False,))
        self.register_attribute('auto', (0,))

        self.n_loops = n_loops
        self.verbose = verbose

        # support standard exported nn~ models:
        # unwrap neutone
        if hasattr(model, 'model'):
            model = model.model
        # unwrap rave+prior
        if hasattr(model, '_rave'):
            model = model._rave
        self.block_size = model.encode_params[3].item()
        try:
            self.sampling_rate = model.sampling_rate.item()
        except AttributeError:
            self.sampling_rate = model.sr
        self.n_latent = model.encode_params[2].item()
        self.model = model

        print(f'{self.block_size=}, {self.sampling_rate=}')

        assert sr==0 or sr==self.sampling_rate

         # pad the limit_margin argument and convert to tensor
        n_fill = self.n_latent - len(limit_margin)
        if n_fill > 0:
            limit_margin = limit_margin + [limit_margin[-1]]*n_fill
        elif n_fill < 0:
            raise ValueError(
                f'{len(limit_margin)=} is greater than {self.n_latent=}')
        limit_margin_t = torch.tensor(limit_margin)

         # pad the latent_signs argument and convert to tensor
        n_fill = self.n_latent - len(latent_signs)
        if n_fill > 0:
            latent_signs = latent_signs + [latent_signs[-1]]*n_fill
        elif n_fill < 0:
            raise ValueError(
                f'{len(latent_signs)=} is greater than {self.n_latent=}')
        latent_signs_t = torch.tensor(latent_signs)[...,None]

        self.register_buffer('latent_signs', latent_signs_t)

        # create Loops
        # loop 0 is the input feature extractor
        self.loops = nn.ModuleList(Loop(
            i, n_loops, n_context, self.n_latent, 
            limit_margin_t, verbose,
        ) for i in range(n_loops+1))

        self.register_buffer(
            'needs_reset', torch.zeros(n_loops+1, dtype=torch.bool))
        self.register_buffer(
            'needs_start', torch.zeros(n_loops+1, dtype=torch.bool))
        self.register_buffer(
            'needs_end', torch.zeros(n_loops+1, dtype=torch.bool))

        self.register_buffer('mask', 
            torch.empty(2, n_loops+1, requires_grad=False))
    
        # auto trigger
        self.register_buffer("zs", torch.zeros(1, n_loops+1, self.n_latent))
        self.register_buffer("landmark_z", torch.zeros(self.n_latent))
        
        self.reset()

        block_size = model.encoder.encoder.downsample_factor
        if model.pqmf is not None: 
            block_size = block_size * model.pqmf.n_band

        ## nn~ methods
        with torch.no_grad():
            self.register_method(
                "encode",
                in_channels=1,
                in_ratio=1,
                out_channels=self.n_latent,
                out_ratio=self.block_size,
                input_labels=['(signal) input'],
                output_labels=['(signal) latent channel %d'%d for d in range(1, self.n_latent+1)], 
            )

            self.register_method(
                "forward",
                in_channels=1,
                in_ratio=1,
                out_channels=n_loops,
                out_ratio=1,
                input_labels=['(signal) input'],
                output_labels=['(signal) loop channel %d'%d for d in range(1, self.n_loops+1)], 
                test_buffer_size=block_size,
            )

            self.register_method(
                "forward_with_latents",
                in_channels=1,
                in_ratio=1,
                out_channels=n_loops*2,
                out_ratio=1,
                input_labels=['(signal) input'],
                output_labels=
                    ['(signal) loop channel %d'%d for d in range(1, self.n_loops+1)]
                    + ['(signal) latents channel %d'%d for d in range(1, self.n_loops+1)],
                test_buffer_size=block_size
            )

    # loop_index has to be a tuple for nn~ reasons
    @torch.jit.export
    def get_loop_index(self) -> int:
        return self.loop_index[0]
    @torch.jit.export
    def set_loop_index(self, i: int) -> int:
        """
        set the current active loop, possibly triggering start/finalize
        negative values cause reset
        """
        i_prev = self.loop_index[0]
        if abs(i) > self.n_loops:
            if self.verbose>0:
                print(f'loop {i} out of range')
            i = 0
        if i!=i_prev:
            if i < 0:
                self.needs_reset[-i] = True
            if i_prev > 0:
                self.needs_end[i_prev] = True
            if i > 0:
                self.needs_start[i] = True
        self.prev_loop_index = i_prev
        self.loop_index = i,
        return 0
    
    @torch.jit.export
    def get_thru(self) -> bool:
        return self.thru[0]
    @torch.jit.export
    def set_thru(self, val: bool) -> int:
        self.thru = val,
        return 0
    
    @torch.jit.export
    def get_auto(self) -> int:
        return self.auto[0]
    @torch.jit.export
    def set_auto(self, val: int) -> int:
        self.auto = val,
        return 0

    @torch.jit.export
    def reset(self):
        self.record_length = 0
        self.step = 0
        self.set_loop_index(0)
        self.set_loop_index(0) # twice to set prev

        for l in self.loops:
            l.reset()

        self.mask.zero_()

        for loop in self.loops:
            loop.feed(torch.zeros(self.n_latent))

    def forward(self, x):
        # print(x.shape)
        x, z = self.process(x)
        return x
    
    @torch.jit.export
    def forward_with_latents(self, x):
        # print('forward_with_latents')
        x, z = self.process(x)
        # print(z[0,:,0])
        # dump latents into audio stream
        # start with 0 and n, to trigger
        z_pad = torch.zeros_like(x)
        z_pad[:,:,1] = z.shape[2]
        z_pad[:,:,2:z.shape[-1]+2] = z
        # concatenate channels
        return torch.cat((x,z_pad), 1)

    @torch.jit.export
    def process(self, x):
        """
        Args:
            x: input audio
                Tensor[1, 1, sample]
        Attributes:
            loop_index: loop record index
                0 for no loop, 1-indexed loop to record, negative index to erase
            thru:
                0 to mute loop while it is being recorded 
                1 to pass the reconstruction through while recording
            auto:
                0 for manual loop control
                1 for replace median similar loop
        Returns:
            audio frame: Tensor[1, loop, sample]
            latent frame: Tensor[1, loop, latent]
        """

        batch = x.shape[0]
        if batch > 1:
            x = x.mean(0)[None]
            print('WARNING: LivingLooper: batching not supported')

        self.step += 1
        if self.verbose > 1:
            print(f'---step {self.step}---')

        # return self.decode(self.encode(x)) ### DEBUG

        # always encode for cache, even if result is not used
        if self.verbose > 1:
            print(f'encoding')
        with torch.no_grad():
            z = self.encode(x).squeeze()
            # print('encoded', z.shape)
        # if self.verbose > 1:
        #     print(f'done')
            # print(z)

        if self.get_auto()>0:
            ### internal auto triggering : (use z to override i)
            # this would be better normalized by KLD somehow?
            zd = torch.linalg.vector_norm(z[:2] - self.landmark_z[:2]).item()
            if (
                z[0].abs().item() > 1
                and zd > 2
                and (self.record_length > 48 * 2048//self.block_size or self.needs_reset.any())
                and torch.rand((1,)).item() > 0.5 ### TEST
                ):
                # auto-set a new loop index

                # choose the (other) loop with current medium-similar z
                others = self.zs[0].clone()
                loop_index = self.get_loop_index()

                if self.get_auto()==1:
                    others[0] = np.inf
                    if loop_index>=0:
                        others[loop_index] = np.inf
                    self.set_loop_index(int(
                        torch.linalg.vector_norm(others - z, 2, -1)
                        .argmin().item()))
                elif self.get_auto()==2:
                    others[0] = z
                    if loop_index>=0:
                        others[loop_index] = z
                    self.set_loop_index(int(
                        torch.linalg.vector_norm(others - z, 2, -1)
                        .median(0).indices.item()))
                else:
                    others[0] = z
                    if loop_index>=0:
                        others[loop_index] = z
                    self.set_loop_index(int(
                        torch.linalg.vector_norm(others - z, 2, -1)
                        .argmax().item()))
                # i = int(torch.randint(0,self.n_loops+1,(1,)).item())
                # print(zd, i)
                # print(z)
                self.landmark_z[:] = z
            ### end auto triggering
                
        loop_index = self.get_loop_index()

        # feed the current input
        # it goes to loop 0, which maintains a continuous loop feature
        # but also to the currently fitting loop
        # for i,loop in enumerate(self.loops):
        #     if i==0 or i==loop_index:
        #         loop.feed(z)

        for i,b in enumerate(self.needs_end):
            if b:
                self.finalize_loop(i)

        for i,b in enumerate(self.needs_reset):
            if b:
                self.reset_loop(i)

        for i,b in enumerate(self.needs_start):
            if b:
                self.record_length = 0
                self.reset_loop(i)
                self.start_loop(i)
                if self.get_thru():
                    # print('unmask')
                    self.mask[1,i] = 1

        self.zs[:,0] = z

        feat = self.get_feature()

        # fit active loops, then predict other loops, then feed
        for l,loop in enumerate(self.loops):
        # for l,loop in enumerate(self.loops[1:],1):
            # NOTE: slicing self.loops is BUGGED here in torchscript?
            # print(l, id(loop), loop.index)
            if l>0:
                if loop_index==l:
                    if self.verbose>1:
                        print(f'\tfitting loop {l}')
                    # feat = self.get_feature(l)
                    loop.partial_fit(self.step, feat, z)
                    self.zs[:,l] = z
                else:
        # for l,loop in enumerate(self.loops):
            # if l>0 and loop_index!=l:
                    if self.verbose>1:
                        print(f'\tpredicting loop {l}')
                    # feat = self.get_feature(l)
                    self.zs[:,l] = loop.predict(self.step, feat)

        for l,loop in enumerate(self.loops):
            loop.feed(self.zs[0,l])

        self.record_length += 1

        # DEBUG
        # self.zs[:] = z
        # self.mask[:,0] = 1
        # self.mask[:,1:] = 0

        if self.verbose>1:
            print(f'decoding')
        with torch.no_grad():
            y = self.decode(self.zs.permute(1,2,0)[1:]) # loops, channels (1), time
        if self.verbose>1:
            print(f'done')

        fade = torch.linspace(0,1,y.shape[2])
        mask = self.mask[1,:,None,None] * fade + self.mask[0,:,None,None] * (1-fade)
        y = y * mask[1:]
        self.mask[0] = self.mask[1]

        y = y.permute(1,0,2) # 1, loop, time
        z = self.zs[:,1:] # 1, loop, latent
        y = y.expand(batch, y.shape[1], y.shape[2])
        z = z.expand(batch, z.shape[1], z.shape[2])
        return y, z

        # print(f'{self.loop_length}, {self.record_index}, {self.loop_index}')

        # return torch.stack((
        #     (torch.rand(self.block_size)-0.5)/3, 
        #     torch.zeros(self.block_size),
        #     (torch.arange(0,self.block_size)/128*2*np.pi).sin()/3 
        # ))[:,None]

    def get_feature(self):#, i:int):
        # """
        # Args:
        #     i: zero indexed target loop number
        # """
        # f = torch.cat((
        #     self.loops[1].get_feature(),
        #     self.loops[2].get_feature(),
        #     self.loops[3].get_feature(),
        #     self.loops[4].get_feature(),
        # ))
        f = torch.cat([loop.get_feature() for loop in self.loops[1:]])
        # k = f.shape[0]
        # print(f[::k//4])
        # print([x.mean().item() for x in f.chunk(4)])
        return f
        # f = torch.cat([loop.get_feature() for loop in self.loops])
        # return f[f.shape[0]//5:]

    def reset_loop(self, i:int):
        """
        call reset on a loop and mute it
        Args:
            i: zero indexed loop
        """
        for j,loop in enumerate(self.loops): 
            if i==j:
                if self.verbose>0:
                    print(f'resetting {i}')
                loop.reset()
                # if self.verbose>0:
                    # print(f'done')
        self.mask[1,i] = 0.
        self.needs_reset[i] = False

    def start_loop(self, i:int):
        for j,loop in enumerate(self.loops): 
            if i==j:
                if self.verbose>0:
                    print(f'starting {i}...')
                loop.replace(self.loops[0])
                # replace the target loop's feature retroactively with the input feature
        self.needs_start[i] = False

    def finalize_loop(self, i:int):
        """
        assemble dataset, fit a loop, unmute
        Args:
            i: zero indexed loop
        """
        if self.verbose > 0:
            print(f'finalize {i}')
        # work around quirk of torchscript
        # (can't index ModuleList except with literal)
        for j,loop in enumerate(self.loops): 
            if i==j:
                loop.finalize()
                    
        self.mask[1,i] = 1.
        self.needs_end[i] = False

    def encode(self, x):
        """
        feature encoder
        """
        # x = x + torch.randn_like(x)*1e-5
        z = self.model.encode(x)
        # TODO -- use temp when available
        # z = self.model.encode(x, temp=0.0)

        z = z*self.latent_signs
        # print(z)
        # return z.clip(-10, 10)
        # return z.clip(-100, 100)
        return z.clip(-1e6, 1e6)

    def decode(self, z):
        """
        audio decoder
        """
        z = z*self.latent_signs
        return self.model.decode(z)
    

def main(
    # encoder-decoder torchscript
    model,
    # run smoke tests before exporting
    test=True,
    # audio sample rate -- currently must match model if given
    sr=0,
    # predictive context size
    context=64,
    # number of loops
    loops=4,
    # latent-limiter feature
    # last value is repeated for remaining latents
    # limit_margin=[0.1, 0.5, 1],
    limit_margin=[0.1, 0.2, 0.5],
    latent_signs=[1],
    # included in output filename
    name="test",
    verbose=0,
):
    logging.basicConfig(
        level=logging.INFO,
        format=colored("[%(relativeCreated).2f] ", "green") +
        "%(message)s")

    if model is not None:
        logging.info("loading encoder-decoder model from checkpoint")
        logging.info(f"using {model}")

        model = torch.jit.load(model).eval()
        try:
            model = model.w2w_base
        except:
            pass
        try:
            model = model.model
        except:
            pass
        # print(model.latent_mean)
    else:
        raise ValueError

    logging.info("creating looper")
    # ls = None if args.LATENT_SIZE is None else int(args.LATENT_SIZE)
    # print(f'{args.LIMIT_MARGIN=}')
    looper = LivingLooper(
        model, 
        loops,
        context, 
        sr,
        limit_margin,
        latent_signs,
        verbose
        )
    looper.eval()

    # smoke test
    def feed(i):
        x = (torch.rand(1, 1, looper.block_size)-0.5)*0.01
        # x = torch.zeros(1, 1, looper.block_size)
        # x = (torch.rand(1, 1, 2**11)-0.5)*0.1
        looper.set_loop_index(i)
        looper(x)

    def smoke_test():
        looper.reset()
        for l in tqdm([0] + [2]*31 + [1]*(context+3) + [0]*10):
            feed(l)
        if test <= 1: return
        #...
        feed(0)

    # with torch.inference_mode():
    logging.info("smoke test with pytorch")
    if test > 0:
        smoke_test()

    looper.reset()

    logging.info("compiling torchscript")
    looper = torch.jit.script(looper)

    logging.info("smoke test with torchscript")
    if test > 0:
        smoke_test()

    looper.reset()
    ###

    fname = f"ll_{name}_l{loops}_z{looper.n_latent}.ts"
    logging.info(f"saving '{fname}'")
    looper.save(fname)

if __name__  == '__main__':
    Fire(main)