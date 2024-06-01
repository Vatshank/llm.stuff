import torch
import cpp_funcs

a, b = 2, 2
cpp_funcs.add(a, b)

a = torch.randn(2, 3)
b = torch.randn(3, 4)

cpp_funcs.mm_torch(a, b)
