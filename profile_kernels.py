import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

import cpp_funcs


def main():
    x = torch.randn(2, 3, 4)
    weight, bias = torch.randn(4), torch.randn(4)
    y = F.layer_norm(x, normalized_shape=[4], weight=weight, bias=bias)

    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                 record_shapes=True, use_cuda=True) as prof:
        with record_function("layernorm"):
            y_hat = cpp_funcs.layernorm_fwd(x, weight, bias)
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("trace_layernorm.json")
    
if __name__ == "__main__":
    main()
