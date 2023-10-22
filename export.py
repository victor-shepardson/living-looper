from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np

from fire import Fire
import logging
from termcolor import colored

class Loop(nn.Module):
    n_loops:int
    context:int
    feature_size:int

    def __init__(self, 
            index:int,
            n_loops:int,
            n_context:int, # maximum time dimension of model feature
            n_fit:int, # maximum dataset size to fit
            n_latent:int,
            limit_margin:Tensor
        ):

        self.index = index
        self.n_loops = n_loops
        self.max_n_context = n_context # now a maximum
        # self.n_memory = n_memory
        self.n_fit = n_fit
        self.n_latent = n_latent


        max_n_feature = n_loops * n_context * n_latent

        super().__init__()
        self.register_buffer('limit_margin', limit_margin)

        self.register_buffer('weights', 
            torch.empty(max_n_feature, n_latent, requires_grad=False))
        self.register_buffer('center', 
            torch.empty(max_n_feature, requires_grad=False))
        self.register_buffer('scale', 
            torch.empty(max_n_feature, requires_grad=False))
        self.register_buffer('bias', 
            torch.empty(n_latent, requires_grad=False))
        self.register_buffer('z_min', 
            torch.empty(n_latent, requires_grad=False))
        self.register_buffer('z_max', 
            torch.empty(n_latent, requires_grad=False))

    def reset(self):
        # self.end_step = 0
        # self.length = 0
        self.context = 0
        self.feature_size = 0

        # self.memory.zero_()
        self.weights.zero_()
        self.bias.zero_()
        self.center.zero_() # feature mean
        self.scale.fill_(1.) # feature std
        self.z_min.fill_(-torch.inf)
        self.z_max.fill_(torch.inf)

    def feat_process(self, x, fit:bool=False):
        fs = x.shape[1]#self.feature_size

        # x = (x/3).tanh()

        if fit:
            c = self.center[:fs] = x.mean(0)
            s = self.scale[:fs] = x.std(0) + 1e-2
        else:
            c = self.center[:fs]
            s = self.scale[:fs]

        # x = (x - c) / s
        x = (x - c)
        x = (x/2).tanh()
        return x

    def target_process(self, z):
        # return z**2
        s = 1
        z = z / s
        z = torch.where(
            z > 1, ((z+1)/2)**2, torch.where(
                z < -1, -((1-z)/2)**2, z))
        return z * s

    def target_process_inv(self, z):
        # return z.sign()*z.abs().sqrt()
        s = 1
        z = z / s
        z =  torch.where(
            z > 1, 2*z**0.5 - 1, torch.where(
                z < -1, 1 - 2*(-z)**0.5, z))
        return z * s

    def fit(self, feature, z):
        """
        Args:
            feature: Tensor[batch, context, loop, latent]
            z: Tensor[batch, latent]
        """
        # print(torch.linalg.vector_norm(feature, dim=(1,2,3)))
        # print(torch.linalg.vector_norm(feature, dim=(0,2,3)))
        # print(torch.linalg.vector_norm(feature, dim=(0,1,3)))
        # print(torch.linalg.vector_norm(feature, dim=(0,1,2)))

        self.context = feature.shape[1]
        assert feature.shape[2]==self.n_loops
        assert feature.shape[3]==self.n_latent
        print("[batch, context, loop, latent]: ", feature.shape)

        feature = feature.reshape(feature.shape[0], -1)
        self.feature_size = fs = feature.shape[1]
        
        z = self.target_process(z)

        feature = self.feat_process(feature, fit=True)

        # feature = feature + torch.randn_like(feature)*1e-7

        b = z.mean(0)
        self.bias[:] = b

        self.z_min[:] = z.min(0).values
        self.z_max[:] = z.max(0).values

        r = torch.linalg.lstsq(feature, z-b, driver='gelsd')
        w = r.solution
        self.weights[:fs] = w

        # print(torch.linalg.vector_norm(feature, dim=1))
        # print(w.norm())
        # print(torch.linalg.matrix_rank(feature))
        print("rank:", r.rank)
        # print(r.solution.shape, r.residuals, r.rank, r.singular_values)
        # print(feature.shape, (z-b).shape)
        # print(w.norm(), feature.norm(), (z-b).norm(), z-b)


    def eval(self, feature):
        """
        Args:
            feature: Tensor[context, loop, latent]
        Returns:
            Tensor[latent]
        """
        feature = feature[-self.context:].reshape(1,-1) 
        # 1 x (loop,latent,ctx)
        fs = feature.shape[1]#self.feature_size
        assert self.feature_size==0 or fs==self.feature_size

        w, b = self.weights[:fs], self.bias

        z = self.feat_process(feature) @ w + b

        z = self.target_process_inv(z).squeeze(0)

        z = z.clamp(self.z_min-self.limit_margin, self.z_max+self.limit_margin)

        return z
    

class LivingLooper(nn.Module):
    __constants__ = ['loops']

    trained_cropped:bool
    sampling_rate:int
    block_size:int
    n_memory:int

    loop_index:int
    prev_loop_index:int
    record_index:int
    step:int
    record_length:int

    latency_correct:int

    def __init__(self, 
            model:torch.jit.ScriptModule, 
            n_loops:int, 
            n_context:int, # maximum time dimension of model feature
            n_fit:int, # maximum dataset size to fit
            latency_correct:int, # in latent frames
            sr:int, # sample rate
            limit_margin:List[float] # limit latents relative to recorded min/max
            ):
        super().__init__()

        self.n_loops = n_loops
        self.max_n_context = n_context # now a maximum
        self.n_fit = n_fit
        self.n_memory = n_fit + n_context + latency_correct + 1 #n_memory

        self.min_loop = 2
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
            raise ValueError(f'{len(limit_margin)=} is greater than {self.n_latent=}')
        limit_margin_t = torch.tensor(limit_margin)

        self.loops = nn.ModuleList(Loop(
            i, n_loops, n_context, n_fit, self.n_latent, limit_margin_t
        ) for i in range(n_loops))

        # continuously updated last N frames of memory
        self.register_buffer('memory', 
            torch.empty(self.n_memory, n_loops, self.n_latent, requires_grad=False))

        self.register_buffer('mask', 
            torch.empty(2, n_loops, requires_grad=False))
    
        # auto trigger
        self.register_buffer("zs", torch.zeros(1, n_loops, self.n_latent))
        self.register_buffer("landmark_z", torch.zeros(self.n_latent))
        
        self.reset()

    @torch.jit.export
    def reset(self):
        self.record_length = 0
        self.step = 0
        self.loop_index = 0
        self.prev_loop_index = 0
        self.record_index = 0

        for l in self.loops:
            l.reset()

        self.memory.zero_()
        self.mask.zero_()

    def forward(self, loop_index:int, x, oneshot:int=0, auto:int=1):
        """
        Args:
            loop_index: loop record index
                0 for no loop, 1-indexed loop to record, negative index to erase
            x: input audio
                Tensor[1, 1, sample]
            oneshot: loop mode 
                0 for continuation, 1 to loop the training data
            auto:
                0 for manual loop control
                1 for replace median similar loop
        Returns:
            audio frame: Tensor[loop, 1, sample]
            latent frame: Tensor[loop, 1, latent]
        """
        self.step += 1
        # return self.decode(self.encode(x)) ### DEBUG

        # always encode for cache, even if result is not used
        z = self.encode(x).squeeze()

        if (loop_index!=self.prev_loop_index and loop_index < 0):
            self.reset_loop(-loop_index - 1)

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
                and (self.record_length > 48 or self.loop_index < 1)
                and torch.rand((1,)).item() > 0.5 ### TEST
                ):
                # auto-set a new loop index

                # choose the (other) loop with current medium-similar z
                idx = self.loop_index-1
                others = self.zs[0].clone()

                if auto==1:
                    if idx>=0:
                        others[idx] = np.inf
                    i = 1+int(torch.linalg.vector_norm(others - z, 2, -1).argmin().item())
                elif auto==2:
                    if idx>=0:
                        others[idx] = z
                    i = 1+int(torch.linalg.vector_norm(others - z, 2, -1).median(0).indices.item())
                else:
                    if idx>=0:
                        others[idx] = z
                    i = 1+int(torch.linalg.vector_norm(others - z, 2, -1).argmax().item())


                # i = int(torch.randint(0,self.n_loops+1,(1,)).item())
                print(zd, i)
                # print(z)
                self.landmark_z[:] = z
            else:
                # previous value persists
                i = self.loop_index
            ### 

        if abs(i) > self.n_loops:
            print(f'loop {i} out of range')
            i = 0

        i_prev = self.loop_index
        # print(i, i_prev, self.loop_length)
        if i!=i_prev: # change in loop select control
            if i_prev > 0: # previously on a loop
                if self.record_length >= self.min_loop: # and it was long enough
                    # convert to 0-index here
                    self.fit_loop(i_prev-1, oneshot)
            if i>0: # starting a new loop recording
                self.record_length = 0
                self.mask[1,i-1] = 0.
            # if i<0: # erasing a loop # now above
                # self.reset_loop(-1-i)
            self.loop_index = i

        # advance record head
        self.advance()

        # store encoded input to current loop
        if i>0:
            # print(x.shape)
            # slice on LHS to appease torchscript
            # z_i = z[0,:,0] # remove batch, time dims
            # inputs have a delay through the soundcard and encoder,
            # so store them latency_correct frames in the past
            for dt in range(self.latency_correct, -1, -1):
                # this loop is fudging it to acommodate latency_correct > 1
                # while still allowing recent context in LL models...
                # could also use a RAVE prior to get another frame here?
                self.record(z, i-1, dt)
            # self.record(z_i, i, self.latency_correct)

        # eval the other loops
        mem = self.get_frames(self.max_n_context, 1) # ctx x loop x latent

        # print(f'{feature.shape=}')
        for j,loop in enumerate(self.loops):
            # skip the loop which is now recording
            if (i-1)!=j:
                # generate outputs
                z_j = loop.eval(mem)
                # store them to memory
                self.record(z_j, j, 0)

        # update memory
        # if self.loop_index >= 0:
        self.record_length += 1

        self.zs[:] = self.get_frames(1) # time (1), loop, latent
        y = self.decode(self.zs.permute(1,2,0)) # loops, channels (1), time

        fade = torch.linspace(0,1,y.shape[2])
        mask = self.mask[1,:,None,None] * fade + self.mask[0,:,None,None] * (1-fade)
        y = y * mask
        self.mask[0] = self.mask[1]

        return y, self.zs.permute(1,0,2)

        # print(f'{self.loop_length}, {self.record_index}, {self.loop_index}')

        # return torch.stack((
        #     (torch.rand(self.block_size)-0.5)/3, 
        #     torch.zeros(self.block_size),
        #     (torch.arange(0,self.block_size)/128*2*np.pi).sin()/3 
        # ))[:,None]

    # def get_loop(self, i:int) -> Optional[Loop]:
    #     for j,loop in enumerate(self.loops):
    #         if i==j: return loop
    #     return None

    def reset_loop(self, i:int):
        """
        call reset on a loop and mute it
        Args:
            i: zero indexed loop
        """
        for j,loop in enumerate(self.loops): 
            if i==j:
                loop.reset()
        self.mask[1,i] = 0.

    def fit_loop(self, i:int, oneshot:int):
        """
        assemble dataset, fit a loop, unmute
        Args:
            i: zero indexed loop
            oneshot: 0 or 1
        """
        print(f'fit {i+1}')

        lc = self.latency_correct

        # drop the most recent lc frames -- there are no target values
        # (inputs are from the past)
        if oneshot:
            # valid recording length
            rl = min(self.n_memory-lc, self.record_length) 
            # model context length: can be up to the full recorded length
            ctx = min(self.max_n_context, rl) 
            mem = self.get_frames(rl, lc)
            # wrap the final n_context around
            # TODO: wrap target loop but not others?
            train_mem = torch.cat((mem[-ctx:], mem),0)
        else:
            # model context length: no more than half the recorded length
            ctx = min(self.max_n_context, self.record_length//2) 
            # valid recording length
            rl = min(self.n_memory-lc, self.record_length+ctx) 
            mem = self.get_frames(rl, lc)
            # print(rl, lc, mem.shape)
            train_mem = mem

        # limit dataset length to last n_fit frames
        train_mem = train_mem[-(self.n_fit+ctx):]

        # dataset of features, targets
        # unfold time dimension to become the 'batch' dimension,
        # appending a new 'context' dimension
        features = train_mem.unfold(0, ctx, 1) # batch, loop, latent, context
        features = features[:-1] # drop last frame (no following target)
        features = features.permute(0,3,1,2).contiguous() # batch, context, loop, latent
        # drop first ctx frames of target (no preceding features)
        targets = train_mem[ctx:,i,:] # batch x latent

        # work around quirk of torchscript
        # (can't index ModuleList except with literal)
        for j,loop in enumerate(self.loops): 
            if i==j:
                # loop.store(mem, self.step) # NOTE: disabled
                loop.fit(features, targets)
                # rollout predictions to make up latency
                for dt in range(lc,0,-1):
                    mem = self.get_frames(loop.context, dt)
                    z = loop.eval(mem)
                    self.record(z, j, dt-1)
                    
        self.mask[1,i] = 1.


    def advance(self):
        """
        advance the record head
        """
        self.record_index = (self.record_index+1)%self.n_memory

    def record(self, z, i:int, dt:int):
        """
        z: Tensor[latent]
        i: loop index
        dt: how many frames ago to record to
        """
        t = (self.record_index-dt)%self.n_memory
        self.memory[t, i, :] = z

    def get_frames(self, n:int, skip:int=0):
        """
        get contiguous tensor out of ring memory
        """
        end = (self.record_index - skip) % self.n_memory + 1
        begin = (end - n) % self.n_memory
        # print(n, begin, end)
        if begin<end:
            return self.memory[begin:end]
        else:
            return torch.cat((self.memory[begin:], self.memory[:end]))

    def encode(self, x):
        """
        feature encoder
        """
        return self.model.encode(x)

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
    # maximum predictive context size
    context=24,
    # maximum number of frames to fit model on
    fit=150,
    # number of loops
    loops=5,
    # max frames for loop memory
    # MEMORY = 1000
    # latency correction, latent frames
    latency_correct=2,
    # latent-limiter feature
    # last value is repeated for remaining latents
    limit_margin=[0.1, 0.5, 1],
    # included in output filename
    name="test",
):
    logging.basicConfig(
        level=logging.INFO,
        format=colored("[%(relativeCreated).2f] ", "green") +
        "%(message)s")

    if model is not None:
        logging.info("loading encoder-decoder model from checkpoint")
        logging.info(f"using {model}")

        model = torch.jit.load(model).eval()
    else:
        raise ValueError

    logging.info("creating looper")
    # ls = None if args.LATENT_SIZE is None else int(args.LATENT_SIZE)
    # print(f'{args.LIMIT_MARGIN=}')
    looper = LivingLooper(
        model, 
        loops,
        context, 
        fit,
        latency_correct,
        sr,
        limit_margin
        )
    looper.eval()

    # smoke test
    def feed(i, oneshot=None):
        with torch.inference_mode():
            x = (torch.rand(1, 1, 2**11)-0.5)*0.1
            if oneshot is not None:
                looper(i, x, oneshot)
            else:
                looper(i, x)

    def smoke_test():
        looper.reset()
        feed(0)
        for _ in range(31):
            feed(2)
        for _ in range(context+3):
            feed(1, oneshot=1)
        for _ in range(10):
            feed(0)

        if test <= 1: return
        for _ in range(fit+3):
            feed(3)
        feed(0)

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

    fname = f"ll_{name}_l{loops}_z{looper.n_latent}.ts"
    logging.info(f"saving '{fname}'")
    looper.save(fname)

if __name__  == '__main__':
    Fire(main)