#include <torch/extension.h>

torch::Tensor gemm_forward_cuda_new(torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _scales, torch::Tensor _zeros);
torch::Tensor gemm_forward_cuda_new_bf16(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scales,
    torch::Tensor _zeros);