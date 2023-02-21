import torch


def uniform_noise(input, t=0.5):
    """U(-0.5, 0.5)"""
    return torch.empty_like(input).uniform_(-t, t)


def quantize(input, mode="round", mean=None, quant=None):
    if mode == "noise":
        return input + uniform_noise(input)
    else:
        if mean is not None:
            input = input - mean

        if quant is not None:
            input = input / quant

        with torch.no_grad():
            diff = input.round() - input
        
        return input + diff


def adaptiveQuantize(input, mode="round", mean=None, quant=None):
    if mean is not None:
        input = input - mean

    if quant is not None:
        input = input / quant

    if mode == "noise":
        diff = uniform_noise(input)
        diff.requires_grad = False
    else:
        with torch.no_grad():
            diff = input.round() - input

    return input + diff


def scale_quant(input, scale=2**8):
    return quantize(input * scale) / scale


def noise_quant(input):
    return quantize(input, mode='noise')


def gumbel_like(input, eps: float = 1e-20):  # ~Gumbel(0, 1)
    exp = torch.rand_like(input).add_(eps).log_().neg_()
    return exp.add_(eps).log_().neg_()


def gumbel_softmax(logits, tau: float = 1, dim: int = -1):
    """Samples from the `Gumbel-Softmax distribution`. ~Gumbel(logits, tau)"""
    gumbels = (logits + gumbel_like(logits)) / tau
    return gumbels.softmax(dim)


class StochasticGumbelAnnealing(torch.nn.Module):

    def __init__(self, init_tau=0.5, c=0.001, iter=0, eps=1e-5):
        super().__init__()
        self.register_buffer("tau", torch.FloatTensor([init_tau]))
        self.register_buffer("iter", torch.IntTensor([iter]))
        self.c = c
        self.eps = eps

        self.step()

    def extra_repr(self):
        return f"tau={self.tau.item():.3e}, c={self.c:.2e}, iter={self.iter.item()}"

    def forward(self, input):
        bound = torch.stack([input.floor(), input.ceil()], dim=-1).detach_()
        distance = bound.sub(input.unsqueeze(-1)).abs()
        logits = torch.atan(distance.clamp_max(1-self.eps))/(-self.tau)
        weight = gumbel_softmax(logits, self.tau, dim=-1)
        output = (bound * weight).sum(dim=-1)
        return output

    def get_tau(self):
        decay = self.c * (self.iter-700)
        return 0.5*torch.exp(-decay.clamp_min(0)).item()

    def step(self, closure=None):
        value = self.get_tau() if closure is None else closure(self.iter)
        self.tau.data.fill_(value).clamp_min_(1e-12)
        self.iter.data += 1


def random_quant(input, mean=None, p=0.5, quant=None):
    """use `m` method random quantize input with  probability `p`, others use round"""
    idxs = torch.rand_like(input).lt(p).bool()
    round_idx = torch.logical_not(idxs)
    output = torch.empty_like(input)
    # output.masked_scatter_(idxs, m(input.masked_select(idxs)))

    if mean is not None and quant is not None:
        m = mean.masked_select(idxs)
        q = quant.masked_select(idxs)
        output.masked_scatter_(idxs, adaptiveQuantize(input.masked_select(idxs), "noise", m, q) * q + m)

        m = mean.masked_select(round_idx)
        q = quant.masked_select(round_idx)
        output.masked_scatter_(round_idx, adaptiveQuantize(input.masked_select(round_idx), "round", m, q) * q + m)
    elif mean is not None:
        m = mean.masked_select(idxs)
        output.masked_scatter_(idxs, adaptiveQuantize(input.masked_select(idxs), "noise", m) + m)

        m = mean.masked_select(round_idx)
        output.masked_scatter_(round_idx, adaptiveQuantize(input.masked_select(round_idx), "round", m) + m)
    elif quant is not None:
        q = quant.masked_select(idxs)
        output.masked_scatter_(idxs, adaptiveQuantize(input.masked_select(idxs), "noise", quant=q) * q)

        q = quant.masked_select(round_idx)
        output.masked_scatter_(round_idx, adaptiveQuantize(input.masked_select(round_idx), "round", quant=q) * q)
    else:
        output.masked_scatter_(idxs, adaptiveQuantize(input.masked_select(idxs), "noise"))
        output.masked_scatter_(round_idx, adaptiveQuantize(input.masked_select(round_idx), "round"))

    return output
