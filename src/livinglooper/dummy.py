import math

import fire
import nn_tilde
import torch

MAX_BATCH = 8

class DummyAutoencoder(nn_tilde.Module):
    def __init__(self, n_latent, block_size, sr=48000):
        super().__init__()
        self.n_latent = n_latent
        self.block_size = block_size
        self.sr = sr

        self.freqs = torch.linspace(0, 1, self.n_latent+1)[1:]**3 / 2
        self.register_buffer('phase', torch.zeros(self.n_latent))
        self.register_buffer('dphase', torch.zeros(self.n_latent))
        self.register_buffer('amp', torch.zeros(MAX_BATCH, self.n_latent))

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
                "decode",
                in_channels=self.n_latent,
                in_ratio=self.block_size,
                out_channels=1,
                out_ratio=1,
                input_labels=['(signal) latent channel %d'%d for d in range(1, self.n_latent+1)], 
                output_labels=['(signal) input'],
            )

    @torch.jit.export
    def encode(self, x):
        """
        Args:
            x: [batch, 1, t]
        """
        t = x.shape[-1]
        # print(f'encode: {x.shape=}')
        phase = self.phase + torch.arange(t)[:,None]*self.freqs
        self.phase[:] = phase[-1]%1

        filters = (phase * (math.pi*2)).sin() # t x n_latent

        # print(f'encode: {filters.shape=}')

        win = 1 - torch.linspace(0, 2*math.pi, self.block_size).cos()

        # batch, n_latent, t//block_size
        z = (
            win*(x * filters.T).unfold(-1,self.block_size,self.block_size)
            ).mean(-1).abs() 

        # print(f'encode: {z.shape=}')

        return z

    @torch.jit.export
    def decode(self, z):
        """
        Args:
            z: [batch, n_latent, t//block_size]
        """
        b = z.shape[0]
        t = z.shape[-1] * self.block_size
        # print(f'decode: {z.shape=}')
        phase = self.dphase + torch.arange(t)[:,None]*self.freqs
        self.dphase[:] = phase[-1]%1

        sigs = (phase * (math.pi*2)).sin() # t x n_latent
        # print(f'decode: {sigs.shape=}')

        # not correct when processing multiple blocks
        # but not important, it's a dummy 
        # batch, n_latent, t
        amp = torch.lerp(
            self.amp[:b,:,None], z[:,:,-1:], torch.linspace(0,1,t+1)[1:])
        self.amp[:b,:] = amp[:,:,-1]
        # print(f'decode: {amp.shape=}')
         
        return (amp * sigs.T).sum(1, keepdim=True) # batch, 1, t
    
def main(n_latent=16, block_size=2048, name='dummy'):
    m = DummyAutoencoder(n_latent=n_latent, block_size=block_size)
    m = torch.jit.script(m)
    m.encode, m.decode
    m.save(f'{name}.ts')

if __name__=='__main__':
    fire.Fire(main)





