from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np

from fire import Fire
import logging
from termcolor import colored
from tqdm import tqdm

from loop import Loop


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
    # store: (stores features and indexes them by global step)

# a rep can be composed from other reps
# e.g. Tanh . Window

# latency correction:
    # fit with a delay on other loops;
    #   i.e. call get with skip=latency correct for other loops
    # roll out predictions after calling finalize

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

class LivingLooper(nn.Module):
    # __constants__ = ['loops']

    sampling_rate:int
    block_size:int

    loop_index:int
    prev_loop_index:int
    step:int
    record_length:int

    latency_correct:int

    verbose:int

    def __init__(self, 
            model:torch.jit.ScriptModule, 
            n_loops:int, 
            n_context:int, # time dimension of model feature
            latency_correct:int, # in latent frames
            sr:int, # sample rate
            limit_margin:List[float], # limit latents relative to recorded min/max
            verbose:int=0
            ):
        super().__init__()

        self.n_loops = n_loops
        self.verbose = verbose

        # self.min_loop = 2
        self.latency_correct = latency_correct

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

        # create Loops
        # loop 0 is the input feature extractor
        self.loops = nn.ModuleList(Loop(
            i, n_loops, n_context, self.n_latent, latency_correct, 
            limit_margin_t, verbose,
        ) for i in range(n_loops+1))

        self.register_buffer('mask', 
            torch.empty(2, n_loops+1, requires_grad=False))
    
        # auto trigger
        self.register_buffer("zs", torch.zeros(1, n_loops+1, self.n_latent))
        self.register_buffer("landmark_z", torch.zeros(self.n_latent))
        
        self.reset()

    @torch.jit.export
    def reset(self):
        self.record_length = 0
        self.step = 0
        self.loop_index = 0
        self.prev_loop_index = 0

        for l in self.loops:
            l.reset()
            l.store.reset()

        self.mask.zero_()

        for step in range(-2*self.latency_correct, 1):
            for loop in self.loops:
                loop.feed(step, torch.zeros(self.n_latent))

    def forward(self, loop_index:int, x, thru:int=0, auto:int=0):
        """
        Args:
            loop_index: loop record index
                0 for no loop, 1-indexed loop to record, negative index to erase
            x: input audio
                Tensor[1, 1, sample]
            thru:
                0 to mute loop while it is being recorded 
                1 to pass the reconstruction through while recording
            auto:
                0 for manual loop control
                1 for replace median similar loop
        Returns:
            audio frame: Tensor[loop, 1, sample]
            latent frame: Tensor[loop, 1, latent]
        """

        self.step += 1
        if self.verbose > 1:
            print(f'---step {self.step}---')

        # return self.decode(self.encode(x)) ### DEBUG

        # always encode for cache, even if result is not used
        if self.verbose > 1:
            print(f'encoding...')
        with torch.no_grad():
            z = self.encode(x).squeeze()
            # print(z)
        if self.verbose > 1:
            print(f'done')
            # print(z)

        if (loop_index!=self.prev_loop_index and loop_index < 0):
            self.reset_loop(-loop_index)

        # this is the previous *input* not necessarily previous actual
        self.prev_loop_index = loop_index

        if auto==0:
            # use the input loop index
            i = loop_index
        else:
            ### internal auto triggering : (use z to override i)
            zd = torch.linalg.vector_norm(z[:2] - self.landmark_z[:2]).item()
            if (
                z[0].abs().item() > 1
                and zd > 3
                and (self.record_length > 48 * 2048//self.block_size or self.loop_index < 1)
                and torch.rand((1,)).item() > 0.5 ### TEST
                ):
                # auto-set a new loop index

                # choose the (other) loop with current medium-similar z
                idx = self.loop_index
                others = self.zs[0].clone()

                if auto==1:
                    others[0] = np.inf
                    if idx>=0:
                        others[idx] = np.inf
                    i = int(torch.linalg.vector_norm(others - z, 2, -1).argmin().item())
                elif auto==2:
                    others[0] = z
                    if idx>=0:
                        others[idx] = z
                    i = int(torch.linalg.vector_norm(others - z, 2, -1).median(0).indices.item())
                else:
                    others[0] = z
                    if idx>=0:
                        others[idx] = z
                    i = int(torch.linalg.vector_norm(others - z, 2, -1).argmax().item())

                # i = int(torch.randint(0,self.n_loops+1,(1,)).item())
                # print(zd, i)
                # print(z)
                self.landmark_z[:] = z
            else:
                # previous value persists
                i = self.loop_index
            ### end auto triggering

        if abs(i) > self.n_loops:
            if self.verbose>0:
                print(f'loop {i} out of range')
            i = 0

        # print(f'active loop {i}')
        i_prev = self.loop_index
        # if i!=i_prev: # change in loop select control
            # print(f'switch from {i_prev}')

        # feed the current input
        # TODO:
        # if just switched from another loop,
        # the previous active loop needs to roll out,
        # and the new active loop already has a prediction,
        # so feed the previous loop instead?
        # alternatively, require the ability for features to 'roll back' one step?
        # or, have a separate rep/store for the input(s)?
        for l,loop in enumerate(self.loops):
            if l in (0, i):
                loop.feed(self.step-self.latency_correct, z)

        # print(i, i_prev, self.loop_length)
        if i!=i_prev: # change in loop select control
            if i_prev > 0: # previously on a loop
                # if self.record_length >= self.min_loop: # and it was long enough
                # convert to 0-index here
                self.finalize_loop(i_prev)
            if i>0: # starting a new loop recording
                self.record_length = 0
                self.reset_loop(i)
                self.start_loop(i)
                if thru:
                    # print('unmask')
                    self.mask[1,i] = 1
            self.loop_index = i

        # fit active loop / predict other loops
        for l,loop in enumerate(self.loops):
        # for l,loop in enumerate(self.loops[1:],1):
            # NOTE: slicing self.loops is BUGGED here in torchscript?
            # print(l, id(loop), loop.index)
            if l > 0:
                if i==l:
                    if self.verbose>1:
                        print(f'--fitting loop {l}--')
                    t = self.step - self.latency_correct
                    feat = self.get_feature(l, self.latency_correct + 1)
                    zl = z
                    loop.partial_fit(t, feat, z)
                    # in this case, feed happened above
                else:
                    if self.verbose>1:
                        print(f'--predicting loop {l}--')
                    feat = self.get_feature(l, 1)
                    zl = loop.predict(self.step, feat)
                    loop.feed(self.step, zl)
                self.zs[:,l] = zl
                # print('done')

        self.record_length += 1

        # DEBUG
        # self.zs[:] = z
        # self.mask[:,0] = 1
        # self.mask[:,1:] = 0

        if self.verbose>1:
            print(f'--decoding...--')
        with torch.no_grad():
            y = self.decode(self.zs.permute(1,2,0)[1:]) # loops, channels (1), time
        if self.verbose>1:
            print(f'done')

        fade = torch.linspace(0,1,y.shape[2])
        mask = self.mask[1,:,None,None] * fade + self.mask[0,:,None,None] * (1-fade)
        y = y * mask[1:]
        self.mask[0] = self.mask[1]


        return y, self.zs[:,1:].permute(1,0,2)

        # print(f'{self.loop_length}, {self.record_index}, {self.loop_index}')

        # return torch.stack((
        #     (torch.rand(self.block_size)-0.5)/3, 
        #     torch.zeros(self.block_size),
        #     (torch.arange(0,self.block_size)/128*2*np.pi).sin()/3 
        # ))[:,None]

    def get_feature(self, i:int, delay:int):
        """
        Args:
            i: zero indexed target loop number
            delay: number of frames in the past
        """
        step = self.step - delay
        return torch.cat([loop.get(
            step if (l==i) else 
            (step - self.latency_correct + 1)
            ) for l,loop in enumerate(self.loops[1:], 1)
        ])

    def reset_loop(self, i:int):
        """
        call reset on a loop and mute it
        Args:
            i: zero indexed loop
        """
        for j,loop in enumerate(self.loops): 
            if i==j:
                if self.verbose>0:
                    print(f'resetting {i}...')
                loop.reset()
                if self.verbose>0:
                    print(f'done')
        self.mask[1,i] = 0.

    def start_loop(self, i:int):
        for j,loop in enumerate(self.loops): 
            if i==j:
                if self.verbose>0:
                    print(f'starting {i}...')
                # replace the target loop's feature retroactively with the input feature
                loop.replace(self.loops[0])

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
                # rollout predictions to make up latency
                for dt in range(self.latency_correct+1,1,-1):
                    # print(f'rollout {dt}')
                    feat = self.get_feature(i,dt)
                    z = loop.predict(self.step-dt, feat)
                    loop.feed(self.step-dt+1, z)
                    
        self.mask[1,i] = 1.

    def encode(self, x):
        """
        feature encoder
        """
        # return self.model.encode(x)
        return self.model.encode(x, temp=0.0)

    def decode(self, z):
        """
        audio decoder
        """
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
    # latency correction, latent frames
    latency_correct=2,
    # latent-limiter feature
    # last value is repeated for remaining latents
    # limit_margin=[0.1, 0.5, 1],
    limit_margin=[0.1, 0.2, 0.5],
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
        latency_correct,
        sr,
        limit_margin,
        verbose
        )
    looper.eval()

    # smoke test
    def feed(i):
        x = (torch.rand(1, 1, looper.block_size)-0.5)*0.01
        # x = (torch.rand(1, 1, 2**11)-0.5)*0.1
        looper(i, x)

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