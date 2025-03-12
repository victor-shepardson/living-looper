import math
import torch
import nn_tilde

class TFR(nn_tilde.Module):
    def __init__(self, 
            max_batch:int=1, 
            # overlap:int=2,
            block:int=2048,
            n_sines:int=8,
        ):
        super().__init__()

        self.max_batch = max_batch
        # self.overlap = overlap
        self.block = block
        self.n_sines = n_sines

        # TODO: buffer to support overlap
        # self.register_buffer('frames', torch.zeros(
            # max_batch, overlap-1, block//2+1))
        # n = (block//2+1)
        n = n_sines
        self.register_buffer('z', torch.zeros(max_batch, n*2, 1))
        self.register_buffer('p', torch.zeros(max_batch, n, 1))
        
        # windows
        t = torch.linspace(0.5, block-0.5, block)*2*math.pi/block
        h = t.cos()*-0.5+0.5
        Dh = t.sin()*0.5 / self.block * 2*math.pi
        self.register_buffer('h', h) # hann window
        self.register_buffer('Dh', Dh) # derivative is sin window
        # self.register_buffer('Th', t*h) # not sure scale of t here

        self.register_buffer('f', torch.linspace(0, math.pi, block//2+1))

    def encode(self, x, eps:float=1e-5):
        """
        Args:
            x: Tensor[batch, channels, n*block]
        """
        b, c, _ = x.shape
        assert c==1, "implement multichannel"
        x = x.reshape(b,c,-1,self.block) # batch, channel, frame, t
        n = x.shape[2]
        x = x + (torch.rand_like(x)-0.5)*eps
        s = torch.fft.rfft(x*self.h, dim=-1, norm='forward')
        # instantaneous frequency
        sf = torch.fft.rfft(x*self.Dh, dim=-1, norm='forward')
        # return s, sf
        w = self.f - (sf / s).imag
        # magnitude
        r = s.abs()
        # smart version: 
        # loop over bins by decreasing power.
            # if nearer to a cluster center than original bin center,
            # add to cluster
            # else make new cluster center
            # should work well under assumption that peaks are close to their
            # original bin centers
        assert b==1, "implement batching for sort"
        steps = []
        for t in range(n):
            centers = []
            powers = []
            # clusters = []
            idx = r[0,0,t,:].argsort(descending=True)
            for i in idx:
                wi = w[0,0,t,i]
                ri = r[0,0,t,i]
                # if ri < eps: break
                dists = [(wc - wi).abs() for wc in centers] 
                if len(centers)<self.n_sines:
                    travel = (self.f[i] - wi).abs()
                    # possibility of creating new clusters
                    dists = dists + [travel]
                # else: NOTE: should be possible to vectorize once all clusters populated
                j = torch.argmin(torch.tensor(dists))
                if j==len(centers):
                    # new cluster
                    centers.append(wi)
                    powers.append(ri)
                    # clusters.append([i])
                else:
                    # add to cluster
                    # could update cluster centers to be weighted mean here
                    # maybe not much advantage though?
                    # could do this across timesteps as a way to preserve cluster ids
                    # clusters[j].append(i)
                    powers[j] += ri
            while len(centers) < self.n_sines:
                centers.append(0)
                powers.append(0)
            z = torch.tensor([*powers, *centers])[None,:]
            steps.append(z)
            # print(centers, powers)

        # dumb version: don't resample, one sine per FFT bin
        # z = torch.cat((r, w), -1)
        # l = z.shape[-1]
        # return z.permute(0,1,3,2).reshape(b,c*l,-1)

        z = torch.stack(steps, -1)
        # z = torch.tensor([*powers, *centers])[None,:,None]
        return z

    def decode(self, z):
        """
        Args:
            z: Tensor[batch, latents, n]
        """
        b, c, n = z.shape
        # cat previous frame
        z = torch.cat((self.z, z), -1)
        # linear interp
        z = torch.nn.functional.interpolate(
            z, (self.block*n+1), mode='linear')[:,:,1:]
        r, w = torch.chunk(z, 2, 1)

        p = self.p + w.cumsum(-1)

        x = (p.sin() * r).sum(1).reshape(b,1,-1)

        self.z[:] = z[:,:,-1:]
        self.p[:] = p[:,:,-1:]

        return x