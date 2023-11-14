import torch
from typing import List
    

# class Chain(torch.nn.Module):
#     def __init__(self, layers:List[torch.nn.Module]):
#         super().__init__()
#         self.ms = torch.nn.ModuleList(layers)
#         for m in self.ms:
#             self.n_feature = m.n_feature
#     def feed(self, x, offset:int=0):
#         for m in self.ms:
#             x = m.feed(x, offset)
#         return x
#     def reset(self):
#         for m in self.ms:
#             m.reset()
#     def replace(self, other:"Chain"):
#         # for m,mo in zip(self.ms, other.ms):
#         #     m.replace(mo)
#         for i,m in enumerate(self.ms):
#             m.replace(other.ms[i])

class Slice(torch.nn.Module):
    def __init__(self, start:int, end:int):
        super().__init__()
        self.start = start
        self.end = end
        self.n_feature = end - start
    def reset(self):
        pass
    def feed(self, x, offset:int=0):
        return x[self.start:self.end]
    def replace(self, other:"Slice"):
        pass
    

class Cat2(torch.nn.Module):
    def __init__(self, a:torch.nn.Module, b:torch.nn.Module):
        super().__init__()
        self.a = a
        self.b = b
        # self.ms = torch.nn.ModuleDict({str(k):m for k,m in enumerate(ms)})
        self.n_feature = a.n_feature + b.n_feature
    def reset(self):
        self.a.reset()
        self.b.reset()
    def feed(self, x, offset:int=0):
        return torch.cat((self.a.feed(x, offset), self.b.feed(x, offset)), -1)
    def replace(self, other:"Cat2"):
        self.a.replace(other.a)
        self.b.replace(other.b)

class Chain2(torch.nn.Module):
    def __init__(self, a:torch.nn.Module, b:torch.nn.Module):
        super().__init__()
        self.a = a
        self.b = b
        # self.ms = torch.nn.ModuleDict({str(k):m for k,m in enumerate(ms)})
        self.n_feature = b.n_feature
    def reset(self):
        self.a.reset()
        self.b.reset()
    def feed(self, x, offset:int=0):
        return self.b.feed(self.a.feed(x, offset), offset)
    def replace(self, other:"Chain2"):
        self.a.replace(other.a)
        self.b.replace(other.b)

def Cat(ms:List[torch.nn.Module]):
    if len(ms)==1:
        return ms[0]
    else:
        return Cat2(ms[0], Cat(ms[1:]))
    
def Chain(ms:List[torch.nn.Module]):
    if len(ms)==1:
        return ms[0]
    else:
        return Chain2(ms[0], Chain(ms[1:]))
# class Cat(torch.nn.Module):
#     def __init__(self, ms:List[torch.nn.Module]):
#         super().__init__()
#         self.n = len(ms)
#         self.ms = torch.nn.ModuleList(ms)
#         # self.ms = torch.nn.ModuleDict({str(k):m for k,m in enumerate(ms)})
#         self.n_feature = sum(m.n_feature for m in ms)
#     def reset(self):
#         for m in self.ms:
#             m.reset()
#     def feed(self, x, offset:int=0):
#         xs = []
#         for m in self.ms:
#             xs.append(m.feed(x, offset))
#         return torch.cat(xs, -1)
#     def replace(self, other:"Cat"):
#         oms:torch.nn.ModuleList = other.ms
#         for i,m in enumerate(self.ms):
#             for j,mo in enumerate(other.ms):
#                 if i==j:
#                     m.replace(mo)
            

class Quadratic(torch.nn.Module):
    def __init__(self, m:torch.nn.Module):
        super().__init__()
        self.m = m
        self.n_feature = m.n_feature**2
    def reset(self):
        self.m.reset()
    def feed(self, x, offset:int=0):
        x = self.m.feed(x, offset)
        return (x * x[...,None]).view(-1)
    def replace(self, other:"Quadratic"):
        self.m.replace(other.m)
    
class DividedQuadratic(torch.nn.Module):
    def __init__(self, m:torch.nn.Module):
        super().__init__()
        self.m = m
        self.n_feature = m.n_feature**2 * 2
    def reset(self):
        self.m.reset()
    def feed(self, x, offset:int=0):
        x = self.m.feed(x, offset)
        return (torch.cat((
            torch.nn.functional.softplus(x), torch.nn.functional.softplus(-x)
            )) * x[...,None]).view(-1)
    def replace(self, other:"DividedQuadratic"):
        self.m.replace(other.m)


class Window(torch.nn.Module):
    """
    """
    record_index:int
    def __init__(self,
        n_target:int,
        n_ctx:int,
        ):
        super().__init__()

        self.record_index = 0
        self.n_ctx = n_ctx
        self.n_memory = n_ctx
        self.register_buffer('memory', torch.empty(self.n_memory, n_target))

        # read by Loop
        self.n_feature = n_ctx*n_target

        self.reset()

    def reset(self):
        self.memory.zero_()

    def replace(self, other:"Window"):
        self.memory[:] = other.memory
        self.record_index = other.record_index

    def wrap(self, idx:int):
        return idx % self.n_memory

    def feed(self, x, offset:int=0):
        """feed a vector into loop memory and update pointer
        Args:
            x: input vector
            offset: number of frames in the past to rewrite
        """
        idx = self.wrap(self.record_index - offset)
        self.memory[idx] = x
        self.record_index = self.wrap(self.record_index+1)
        return self.get(offset)

    def get(self, offset:int=0):
        """read out of loop memory"""
        # end = (self.record_index) % self.n_ctx + 1
        end = self.wrap(self.record_index - 1 - offset) + 1
        begin = self.wrap(end - self.n_ctx)
        # print(n, begin, end)
        if begin<end:
            r = self.memory[begin:end]
        else:
            r = torch.cat((self.memory[begin:], self.memory[:end]))

        return r.reshape(-1)

class RNN(torch.nn.Module):
    """
    """
    record_index:int
    def __init__(self,
        n_target:int,
        n_hidden:int,
        n_out:int,
        ):
        super().__init__()

        self.register_buffer('hidden', torch.empty(3, n_hidden))
        self.register_buffer('ih', torch.empty(n_target, n_hidden))
        self.register_buffer('hh', torch.empty(n_hidden, n_hidden))
        self.register_buffer('ho', torch.empty(n_hidden, n_out))

        # read by Loop
        self.n_feature = n_out

        self.reset()

    def init(self, t):
        t[:] = torch.randn_like(t)
        # t.mul_((torch.rand_like(t) > 0.5).int())
        t.mul_(2**-0.5 / t.pow(2).sum(0).sqrt())

    def reset(self):
        self.hidden.zero_()
        # self.ih[:] = torch.randn_like(self.ih)
        # self.ih = self.ih / self.ih.pow(2).sum(0).sqrt()
        # self.ho[:] = torch.randn_like(self.ho)
        # self.ho = self.ho / self.ho.pow(2).sum(0).sqrt()
        # self.hh[:] = torch.randn_like(self.hh)
        # self.hh = self.hh / self.hh.pow(2).sum(0).sqrt()
        self.init(self.ih)
        self.init(self.hh)
        self.init(self.ho)

    def replace(self, other:"Window"):
        self.hidden[:] = other.hidden

    def wrap(self, idx:int):
        return idx % self.n_memory

    def feed(self, x, offset:int=0):
        """feed a vector into loop memory and update pointer
        Args:
            x: input vector
            offset: number of frames in the past to rewrite
        """
        h = self.hidden[offset]

        h_new = x @ self.ih + h @ self.hh
        h_new = h_new.sin()
        # mix = torch.linspace(0, -5, h.shape[0]).exp()
        # h = mix * h_new + (1-mix) * h

        for i in range(2,offset,-1):
            self.hidden[i] = self.hidden[i-1]
        self.hidden[offset] = h

        return h @ self.ho