import torch
import torch.nn.functional as F

import cpp_funcs


def test_relu():
    x = torch.randn(2, 3, 4)
    y = torch.relu(x)
    y_hat = cpp_funcs.relu_fwd(x)
    assert torch.isclose(y, y_hat).all()
    

def test_layernorm():
    x = torch.randn(2, 3, 4)
    weight, bias = torch.randn(4), torch.randn(4)
    y = F.layer_norm(x, normalized_shape=[4], weight=weight, bias=bias)
    y_hat = cpp_funcs.layernorm_fwd(x, weight, bias)
    assert torch.isclose(y, y_hat).all()

def test_softmax():
    x = torch.randn(2, 3, 3)
    y = F.softmax(x, dim=-1)
    y_hat = cpp_funcs.softmax_fwd(x)
    assert torch.isclose(y, y_hat).all()

def test_matmul():
    A = torch.randn(256, 512)
    B = torch.randn(512, 1024)
    y = torch.matmul(A, B)
    y_hat = cpp_funcs.matmul(A, B)
    print((y - y_hat).abs().max())
    assert torch.isclose(y, y_hat, atol=1e-4).all()

if __name__ == "__main__":
    test_relu()
    test_layernorm()
    test_softmax()
    test_matmul()
    print("All tests passed!")
