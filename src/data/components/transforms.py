from torch import nn


class TransSqueeze(nn.Module):
    def __init__(self, dim=0):
        super(TransSqueeze, self).__init__()

        self.dim = dim

    def __call__(self, x):
        return x.squeeze(self.dim)


class TransUnSqueeze(nn.Module):
    def __init__(self, dim=0):
        super(TransUnSqueeze, self).__init__()

        self.dim = dim

    def __call__(self, x):
        return x.unsqueeze(self.dim)


class TransChunked(nn.Module):
    def __init__(self, chunk=32, width=16):
        super(TransChunked, self).__init__()
        self.chunk = chunk
        self.width = width

    def __call__(self, x):
        num = x.shape[1]//self.width - 1
        l = []
        for i in range(num):
            start = i*self.width
            end = start+self.chunk
            l.append(x[:, start:end])

        l.append(x[:, (x.shape[1]-self.chunk):x.shape[1]])
        return l
