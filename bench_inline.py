import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
torch::Tensor mm_torch(torch::Tensor &A, torch::Tensor &B) {
  torch::Tensor C = torch::mm(A, B);
  std::cout << "Sizes of A, B and C: " << A.sizes() << ", " << B.sizes() << ", " << C.sizes() << std::endl;
  return C;
}
"""

mm_extension = load_inline(
    name='mm_extension',
    cpp_sources=[cpp_source],
    # cuda_sources=cuda_source,
    functions=['mm_torch'],
    verbose=True,
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    build_directory='./tmp',
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)

a = torch.randn(2, 3).cuda()
b = torch.randn(3, 4).cuda()

print(mm_extension.mm_torch(a, b))
