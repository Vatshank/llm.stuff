#include <torch/extension.h>

int add(int a, int b) {
  return a + b;
}

void mm_torch(const torch::Tensor &A, const torch::Tensor &B) {
  torch::Tensor C = torch::mm(A, B);
  std::cout << "Sizes of A, B and C: " << A.sizes() << ", " << B.sizes() << ", " << C.sizes() << std::endl;
}

namespace py = pybind11;

PYBIND11_MODULE(cpp_funcs, m) {
    m.doc() = "pybind11 binding to cpp/cuda stuff";

    m.def("mm_torch", &mm_torch, "Vanilla torch matmul");
    
    m.def("add", &add, "A function which adds two numbers");
}
