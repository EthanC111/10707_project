import torch
from transformers import SGD, AdamW, Adafactor

def make_dp_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            self.accum_grads = []

            for i, param in enumerate(self.params):
                self.accum_grads.append(torch.zeros(*param.shape, device=param.device) if param.requires_grad else None)

        def reset_microbatch_grad(self):
            super(DPOptimizerClass, self).reset_grad()

        def microbatch_step(self):
            total_norm = 0.
            for i, param in enumerate(self.params):
                if param.requires_grad:
                    total_norm += (param.grad ** 2).sum().reshape((1,))
            total_norm = total_norm ** .5

            clip_coef = torch.clamp(self.l2_norm_clip * ((total_norm + 1e-6) ** -1)[0], max=1)

            for i, param in enumerate(self.params):
                if param.requires_grad:
                    self.accum_grads[i] += param.grad * clip_coef

        def reset_grad(self):
            super(DPOptimizerClass, self).reset_grad()
            
            for i, param in enumerate(self.params):
                if self.accum_grads[i] is not None:
                    self.accum_grads[i] = torch.zeros(*param.shape, device=param.device)

        def step(self, *args, **kwargs):
            for i, param in enumerate(self.params):
                if param.requires_grad:
                    param.grad = self.accum_grads[i]
                    param.grad += self.l2_norm_clip * self.noise_multiplier * toorch.randn(*param.shape, device=param.device)
                    param.grad *= self.microbatch_size / self.minibatch_size
            super(DPOptimizerClass, self).step(*args, **kwargs)

    return DPOptimizerClass


def get_optimizer(name, params, lr, weight_decay, DP=False):
    if name == 'Adafactor':
        opt_cls = Adafactor
        args = {"scale_parameter": False, "relative_step": False}
    elif name == 'AdamW':
        opt_cls = AdamW
        args = {}
    if DP:
        opt_cls = make_dp_optimizer_class(opt_cls)

    opt = opt_cls(params, lr=lr, weight_decay=weight_decay, *args)
    return opt

DP_SGD = make_dp_optimizer_class(SGD)
DP_AdamW = make_dp_optimizer_class(AdamW)
DP_Adafactor = make_dp_optimizer_class(DP_Adafactor)
