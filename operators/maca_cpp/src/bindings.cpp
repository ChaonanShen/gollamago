#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>

extern "C" int launch_rms_norm_bf16(const void* input, const void* weight,
                                     void* output, int64_t rows,
                                     int64_t columns, float eps, void* stream);
extern "C" int launch_rope_bf16(const void* input, const void* sin_table,
                                 const void* cos_table, void* output,
                                 int64_t batch, int64_t sequence, int64_t heads,
                                 int64_t head_dim, int block_rows, int threads, void* stream);

namespace {

void check_maca_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on a CUDA-compatible MACA device");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor rms_norm(torch::Tensor input, torch::Tensor weight, double eps) {
  check_maca_tensor(input, "input");
  check_maca_tensor(weight, "weight");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
  TORCH_CHECK(input.size(-1) == weight.numel(),
              "weight size must match the final input dimension");
  TORCH_CHECK(input.device() == weight.device(),
              "input and weight must be on the same device");
  TORCH_CHECK(input.size(-1) > 0, "the normalized dimension must be non-empty");
  TORCH_CHECK(eps >= 0.0, "eps must be non-negative");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty_like(input);
  const int64_t columns = input.size(-1);
  const int64_t rows = input.numel() / columns;
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int status = launch_rms_norm_bf16(
      input.data_ptr(), weight.data_ptr(), output.data_ptr(), rows, columns,
      static_cast<float>(eps), reinterpret_cast<void*>(stream.stream()));
  TORCH_CHECK(status == 0, "MXMACA RMSNorm launch failed with error ", status);
  return output;
}

torch::Tensor rope(torch::Tensor input, torch::Tensor sin_table,
                   torch::Tensor cos_table, int64_t block_rows, int64_t threads) {
  check_maca_tensor(input, "input");
  check_maca_tensor(sin_table, "sin_table");
  check_maca_tensor(cos_table, "cos_table");
  TORCH_CHECK(input.dim() == 4, "input must have shape [batch, sequence, heads, head_dim]");
  TORCH_CHECK(sin_table.dim() == 2 && cos_table.dim() == 2,
              "RoPE tables must be two-dimensional");
  TORCH_CHECK(sin_table.sizes() == cos_table.sizes(),
              "RoPE tables must have the same shape");
  TORCH_CHECK(input.size(1) <= sin_table.size(0), "RoPE tables are too short");
  TORCH_CHECK(input.size(3) % 2 == 0 && sin_table.size(1) == input.size(3) / 2,
              "RoPE table width must equal head_dim / 2");
  TORCH_CHECK(input.device() == sin_table.device() && input.device() == cos_table.device(),
              "RoPE tensors must be on the same device");

  TORCH_CHECK(input.size(3) > 0, "head_dim must be positive");
  TORCH_CHECK(block_rows >= 0 && block_rows <= 1024,
              "block_rows must be 0 (automatic) or in [1, 1024]");
  TORCH_CHECK(threads == 0 || (threads >= 64 && threads <= 1024 && threads % 64 == 0),
              "threads must be 0 (automatic) or a multiple of 64 in [64, 1024]");
  // C500 16 GiB slice: joint row/thread sweep across decode and prefill.
  if (block_rows == 0) block_rows = input.size(3) >= 128 ? 4 : 8;
  if (threads == 0) threads = 256;

  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty_like(input);
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const int status = launch_rope_bf16(
      input.data_ptr(), sin_table.data_ptr(), cos_table.data_ptr(), output.data_ptr(),
      input.size(0), input.size(1), input.size(2), input.size(3),
      static_cast<int>(block_rows), static_cast<int>(threads),
      reinterpret_cast<void*>(stream.stream()));
  TORCH_CHECK(status == 0, "MXMACA RoPE launch failed with error ", status);
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("rms_norm", &rms_norm, "MXMACA BF16 RMSNorm");
  module.def("rope", &rope, "MXMACA BF16 RoPE",
             pybind11::arg("input"), pybind11::arg("sin_table"),
             pybind11::arg("cos_table"), pybind11::arg("block_rows") = 0,
             pybind11::arg("threads") = 0);
}
