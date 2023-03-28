import torch
from transformers.optimization import AdamW, Adafactor

def make_dp_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, microbatch_size, num_microbatches, **kwargs):
            print(kwargs)
            super(DPOptimizerClass, self).__init__(**kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.accum_grads = []
            self.step_counter = 0
            self.num_microbatches = num_microbatches

            for group in self.param_groups:
                for param in group["params"]:
                    self.accum_grads.append(torch.zeros(*param.shape, device=param.device) if param.requires_grad else None)

        def step(self, *args, **kwargs):
            if self.step_counter == self.num_microbatches:
                self.step_super(*args, **kwargs)
                self.zero_grad_super()
                self.step_counter = 0

            else:
                total_norm = 0.
                for group in self.param_groups:
                    for param in group["params"]:
                        if param.requires_grad:
                            total_norm += (param.grad ** 2).sum().reshape((1,))
                total_norm = total_norm ** .5

                clip_coef = torch.clamp(self.l2_norm_clip * ((total_norm + 1e-6) ** -1)[0], max=1)

                i = 0
                for group in self.param_groups:
                    for param in group["params"]:
                        if param.requires_grad:
                            self.accum_grads[i] += param.grad * clip_coef
                        i += 1

                self.step_counter += 1 

        def zero_grad_super(self): 
            self.accum_grads = []          
            for group in self.param_groups:
                for param in group["params"]:
                    self.accum_grads.append(torch.zeros(*param.shape, device=param.device) if param.requires_grad else None)

        def step_super(self, *args, **kwargs):
            i = 0
            for group in self.param_groups:
                for param in group["params"]:
                    if param.requires_grad:
                        param.grad = self.accum_grads[i]
                        param.grad += self.l2_norm_clip * self.noise_multiplier * torch.randn(*param.shape, device=param.device)
                        param.grad *= 1 / self.num_microbatches
                    i += 1
            super(DPOptimizerClass, self).step(*args, **kwargs)

    return DPOptimizerClass


def get_optimizer(name, config, DP_config=None):
    if name == 'Adafactor':
        opt_cls = Adafactor
    elif name == 'AdamW':
        opt_cls = AdamW

    if DP_config is not None:
        opt_cls = make_dp_optimizer_class(opt_cls)
        opt = opt_cls(l2_norm_clip=DP_config['l2_norm_clip'],
            noise_multiplier=DP_config['noise_multiplier'],
            microbatch_size=DP_config['microbatch_size'],
            num_microbatches=DP_config['num_microbatches'],
            **config)
    else:
        opt = opt_cls(params, **config)
    return opt
