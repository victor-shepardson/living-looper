import torch

class LinQuad(torch.nn.Module):
    """
    """
    def __init__(self, thresh=1):
        super().__init__()
        self.thresh = thresh

    def forward(self, z):
        z = z/self.thresh
        return torch.where(
            z > 1, ((z+1)/2)**2, torch.where(
                z < -1, -((1-z)/2)**2, z))

    def inv(self, z):
        z = torch.where(
            z > 1, 2*z**0.5 - 1, torch.where(
                z < -1, 1 - 2*(-z)**0.5, z))
        return z * self.thresh
    
class Quad(torch.nn.Module):
    """
    """
    def forward(self, z):
        return z.sign() * z**2

    def inv(self, z):
        return z.sign()*z.abs().sqrt()
    
class Tanh(torch.nn.Module):
    """
    """
    def forward(self, z):
        return (z/3).tanh()

    def inv(self, z):
        return z.arctanh()*3
    
class Id(torch.nn.Module):
    """
    """
    def forward(self, z):
        return z

    def inv(self, z):
        return z