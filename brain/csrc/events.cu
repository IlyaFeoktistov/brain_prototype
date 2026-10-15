// Плотный список событий шага (C++/CUDA, расширение PyTorch).
//
// Таблица синапсов — CSR: у нейрона строка start..start+count в perm.
// Раньше шаг собирал для каждого места запаса выстреливших (max_spikes)
// строку полной ширины — сотни тысяч ячеек, почти все пустые. Здесь блок
// на выстреливший нейрон копирует только его настоящие синапсы подряд
// (смещения — префиксная сумма длин строк), в буфер постоянной ёмкости:
// форма не меняется — шаг остаётся записанным CUDA-графом.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void compact_events_kernel(
    const int64_t* __restrict__ fired, int64_t slots, int64_t n,
    const int32_t* __restrict__ start, const int32_t* __restrict__ count,
    const int32_t* __restrict__ perm, const int64_t* __restrict__ offsets,
    int64_t capacity, int64_t* __restrict__ synapses, int64_t* __restrict__ pre) {
  const int64_t i = blockIdx.x;
  if (i >= slots) return;
  const int64_t neuron = fired[i];
  if (neuron >= n) return;  // фиктивное место запаса
  const int64_t row = start[neuron];
  const int64_t length = count[neuron];
  const int64_t base = offsets[i];
  for (int64_t j = threadIdx.x; j < length; j += blockDim.x) {
    const int64_t at = base + j;
    if (at >= capacity) return;  // не влезло — считается в dropped снаружи
    synapses[at] = perm[row + j];
    pre[at] = neuron;
  }
}

void compact_events(torch::Tensor fired, int64_t n, torch::Tensor start, torch::Tensor count,
                    torch::Tensor perm, torch::Tensor offsets, torch::Tensor synapses, torch::Tensor pre) {
  const int64_t slots = fired.size(0);
  if (slots == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  compact_events_kernel<<<slots, 128, 0, stream>>>(
      fired.data_ptr<int64_t>(), slots, n, start.data_ptr<int32_t>(), count.data_ptr<int32_t>(),
      perm.data_ptr<int32_t>(), offsets.data_ptr<int64_t>(), synapses.size(0),
      synapses.data_ptr<int64_t>(), pre.data_ptr<int64_t>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compact_events", &compact_events, "Плотный список событий шага");
}
