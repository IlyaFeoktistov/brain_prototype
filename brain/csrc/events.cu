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


// Доставка, LTD и LTP по плотному списку событий — одним ядром каждое.
// Раньше то же делали десятки операций PyTorch над всем буфером ёмкости
// (2% синапсов — запас под залпы при шаге 2 мс): при шаге 0.5 мс средний
// шаг занимает ~10% буфера, и 90% работы шли по пустым ячейкам. Ядро идёт
// только до настоящего числа событий (total — на видеокарте, без
// синхронизации: шаг остаётся CUDA-графом). Счёт тот же, что в engine._step.

#include <cuda_fp16.h>

// Путь синапса — последний, чьё начало не дальше синапса.
__device__ __forceinline__ int64_t path_of(const int64_t* __restrict__ path_start, int64_t paths, int64_t syn) {
  int64_t lo = 0, hi = paths - 1;
  while (lo < hi) {
    const int64_t mid = (lo + hi + 1) >> 1;
    if (path_start[mid] <= syn) lo = mid; else hi = mid - 1;
  }
  return lo;
}

// События шага без промежуточного буфера: блок берёт выстреливший нейрон
// (шагом по сетке), потоки — синапсы его строки таблицы (CSR). Буфера нет —
// переполняться нечему, каждое событие ровно один раз; размер запуска
// постоянный — шаг остаётся CUDA-графом.
// Выстрелившие — в начале списка по возрастанию, дальше заполнитель n:
// блок останавливается на первом заполнителе.
#define FOR_EVENTS(fired, slots, n, start, count, perm)                          \
  for (int64_t i_ = blockIdx.x, neuron_; i_ < (slots) && (neuron_ = (fired)[i_]) < (n); i_ += gridDim.x) \
    for (int64_t j_ = threadIdx.x, len_ = (count)[neuron_], row_ = (start)[neuron_]; \
         j_ < len_; j_ += blockDim.x)

__global__ void deliver_kernel(
    const int64_t* __restrict__ fired, int64_t slots, int64_t n,
    const int32_t* __restrict__ start, const int32_t* __restrict__ count, const int32_t* __restrict__ perm,
    const int64_t* __restrict__ path_post, const int64_t* __restrict__ path_fanin,
    const float* __restrict__ net_weight, const int64_t* __restrict__ path_start, const uint8_t* __restrict__ path_flags, int64_t paths,
    float ach_suppression, const float* __restrict__ acetylcholine,
    const bool* __restrict__ dendrite_pre, const bool* __restrict__ dendrite_post,
    const int8_t* __restrict__ net_branch, int64_t branches,
    const bool* __restrict__ typed_pre, const int64_t* __restrict__ cell_type, int64_t kinds,
    const float* __restrict__ stp_u4, const float* __restrict__ stp_x4, const float* __restrict__ stp_U4,
    const bool* __restrict__ is_lamp5, const float* __restrict__ ngf_d, float lamp5_scale, int64_t lamp5_kind,
    const float* __restrict__ stp_x, const float* __restrict__ inh_gain,
    float* __restrict__ dendrite, float* __restrict__ drive, float* __restrict__ glutamate,
    float* __restrict__ inhibition, float* __restrict__ ngf) {
  const float ach = ach_suppression > 0.0f ? *acetylcholine * ach_suppression : 0.0f;
  FOR_EVENTS(fired, slots, n, start, count, perm) {
    const int64_t syn = perm[row_ + j_];
    const int64_t pre = neuron_;
    // Признаки пути: бит 0 — на дендрит, бит 1 — подавляет ацетилхолин,
    // бит 2 — без истощения пресинапса.
    const int64_t path = path_of(path_start, paths, syn);
    const int64_t post = path_post[path] + (syn - path_start[path]) / path_fanin[path];
    float w = net_weight[syn];
    const uint8_t flags = path_flags[path];
    if (flags & 2) w *= 1.0f - ach;
    // Рабочая память -> смысл: на дальний дендрит (вес до истощения).
    if ((dendrite_pre[pre] && dendrite_post[post]) || (flags & 1)) {
      atomicAdd(dendrite + post * branches + net_branch[syn], w);
      continue;
    }
    const bool typed = typed_pre[pre];
    if (typed) {
      int64_t k = cell_type[post];
      const int64_t slot = pre * kinds + (k < 0 ? 0 : k);
      const float u = stp_u4[slot];
      float gain = u * stp_x4[slot] / fmaxf(1.0f - u, 1e-3f) / stp_U4[slot];
      if (is_lamp5[pre]) gain *= ngf_d[pre] * lamp5_scale;
      w *= gain;
    } else if (!(flags & 4)) {
      // Бит 2 — путь без истощения (другой выход клетки с депрессией пути).
      w *= stp_x[pre];
    }
    if (w < 0.0f) {
      w *= inh_gain[post];
      atomicAdd((cell_type[pre] == lamp5_kind ? ngf : inhibition) + post, w);
    } else if (w > 0.0f) {
      atomicAdd(drive + post, w);
      if (glutamate) atomicAdd(glutamate + post, w);
    }
  }
}

__global__ void ltd_kernel(
    const int64_t* __restrict__ fired, int64_t slots, int64_t n,
    const int32_t* __restrict__ start, const int32_t* __restrict__ count, const int32_t* __restrict__ perm,
    const int64_t* __restrict__ path_start, const int64_t* __restrict__ path_post,
    const int64_t* __restrict__ path_fanin, int64_t paths, const float* __restrict__ trace_post, const float* __restrict__ ltd_mask,
    const float* __restrict__ symmetric_area, float a_plus, float a_minus,
    const float* __restrict__ net_weight, const float* __restrict__ ltd_ref,
    const float* __restrict__ scale, __half* __restrict__ eligibility) {
  const float inv = 1.0f / *scale;
  FOR_EVENTS(fired, slots, n, start, count, perm) {
    const int64_t syn = perm[row_ + j_];
    const int64_t pre = neuron_;
    const int64_t path = path_of(path_start, paths, syn);
    const int64_t post = path_post[path] + (syn - path_start[path]) / path_fanin[path];
    float sign = -a_minus;
    if (symmetric_area) {
      // Симметричное окно: пре после поста — тоже потенциация.
      const float zone = symmetric_area[post];
      if (zone > 0.0f && zone == symmetric_area[pre]) sign = a_plus;
    }
    // LTD пропорциональна весу (Bi & Poo 1998; van Rossum et al. 2000):
    // ltd_ref пути — 1 / средний врождённый вес, 0 — без зависимости.
    if (ltd_ref && sign < 0.0f && ltd_ref[path] > 0.0f) sign *= net_weight[syn] * ltd_ref[path];
    const __half add = __float2half(trace_post[post] * ltd_mask[pre] * (sign * inv));
    eligibility[syn] = __hadd(eligibility[syn], add);  // синапс в списке шага один раз
  }
}

__global__ void ltp_kernel(
    const int64_t* __restrict__ fired, int64_t slots, int64_t n,
    const int32_t* __restrict__ start, const int32_t* __restrict__ count, const int32_t* __restrict__ perm,
    const int32_t* __restrict__ net_pre, const float* __restrict__ ltp_kernel_v, float a_plus,
    const float* __restrict__ scale, __half* __restrict__ eligibility) {
  const float inv = 1.0f / *scale;
  FOR_EVENTS(fired, slots, n, start, count, perm) {
    const int64_t syn = perm[row_ + j_];
    const __half add = __float2half(ltp_kernel_v[net_pre[syn]] * (a_plus * inv));
    eligibility[syn] = __hadd(eligibility[syn], add);
  }
}

template <typename T>
static const T* opt(const c10::optional<torch::Tensor>& t) {
  return t.has_value() ? reinterpret_cast<const T*>(t->data_ptr()) : nullptr;
}

static constexpr int BLOCKS = 1024, THREADS = 256;

void deliver(torch::Tensor fired, int64_t n, torch::Tensor start, torch::Tensor count, torch::Tensor perm,
             torch::Tensor path_post, torch::Tensor path_fanin, torch::Tensor net_weight,
             torch::Tensor path_start, torch::Tensor path_flags, bool ach_paths, double ach_suppression,
             torch::Tensor acetylcholine, torch::Tensor dendrite_pre, torch::Tensor dendrite_post,
             torch::Tensor net_branch, int64_t branches,
             torch::Tensor typed_pre, torch::Tensor cell_type, int64_t kinds,
             torch::Tensor stp_u4, torch::Tensor stp_x4, torch::Tensor stp_U4,
             torch::Tensor is_lamp5, torch::Tensor ngf_d, double lamp5_scale, int64_t lamp5_kind,
             torch::Tensor stp_x, torch::Tensor inh_gain,
             torch::Tensor dendrite, torch::Tensor drive, c10::optional<torch::Tensor> glutamate,
             torch::Tensor inhibition, torch::Tensor ngf) {
  auto stream = at::cuda::getCurrentCUDAStream();
  deliver_kernel<<<BLOCKS, THREADS, 0, stream>>>(
      fired.data_ptr<int64_t>(), fired.size(0), n, start.data_ptr<int32_t>(), count.data_ptr<int32_t>(), perm.data_ptr<int32_t>(),
      path_post.data_ptr<int64_t>(), path_fanin.data_ptr<int64_t>(), net_weight.data_ptr<float>(),
      path_start.data_ptr<int64_t>(), path_flags.data_ptr<uint8_t>(), path_start.size(0),
      ach_paths ? (float)ach_suppression : 0.0f, acetylcholine.data_ptr<float>(),
      dendrite_pre.data_ptr<bool>(), dendrite_post.data_ptr<bool>(),
      net_branch.data_ptr<int8_t>(), branches,
      typed_pre.data_ptr<bool>(), cell_type.data_ptr<int64_t>(), kinds,
      stp_u4.data_ptr<float>(), stp_x4.data_ptr<float>(), stp_U4.data_ptr<float>(),
      is_lamp5.data_ptr<bool>(), ngf_d.data_ptr<float>(), (float)lamp5_scale, lamp5_kind,
      stp_x.data_ptr<float>(), inh_gain.data_ptr<float>(),
      dendrite.data_ptr<float>(), drive.data_ptr<float>(),
      glutamate.has_value() ? glutamate->data_ptr<float>() : nullptr,
      inhibition.data_ptr<float>(), ngf.data_ptr<float>());
}

void ltd(torch::Tensor fired, int64_t n, torch::Tensor start, torch::Tensor count, torch::Tensor perm,
         torch::Tensor path_start, torch::Tensor path_post, torch::Tensor path_fanin,
         torch::Tensor trace_post, torch::Tensor ltd_mask, c10::optional<torch::Tensor> symmetric_area,
         double a_plus, double a_minus, torch::Tensor net_weight, c10::optional<torch::Tensor> ltd_ref,
         torch::Tensor scale, torch::Tensor eligibility) {
  auto stream = at::cuda::getCurrentCUDAStream();
  ltd_kernel<<<BLOCKS, THREADS, 0, stream>>>(
      fired.data_ptr<int64_t>(), fired.size(0), n, start.data_ptr<int32_t>(), count.data_ptr<int32_t>(), perm.data_ptr<int32_t>(),
      path_start.data_ptr<int64_t>(), path_post.data_ptr<int64_t>(), path_fanin.data_ptr<int64_t>(),
      path_start.size(0), trace_post.data_ptr<float>(), ltd_mask.data_ptr<float>(),
      opt<float>(symmetric_area), (float)a_plus, (float)a_minus,
      net_weight.data_ptr<float>(), opt<float>(ltd_ref), scale.data_ptr<float>(),
      reinterpret_cast<__half*>(eligibility.data_ptr()));
}

void ltp(torch::Tensor fired, int64_t n, torch::Tensor start, torch::Tensor count, torch::Tensor perm, torch::Tensor net_pre, torch::Tensor ltp_kernel_v,
         double a_plus, torch::Tensor scale, torch::Tensor eligibility) {
  auto stream = at::cuda::getCurrentCUDAStream();
  ltp_kernel<<<BLOCKS, THREADS, 0, stream>>>(
      fired.data_ptr<int64_t>(), fired.size(0), n, start.data_ptr<int32_t>(), count.data_ptr<int32_t>(), perm.data_ptr<int32_t>(),
      net_pre.data_ptr<int32_t>(), ltp_kernel_v.data_ptr<float>(), (float)a_plus, scale.data_ptr<float>(),
      reinterpret_cast<__half*>(eligibility.data_ptr()));
}

// Прежняя доставка через буфер событий (compact_events) — только для
// сверки новой доставки по строкам (tools/kernel_check.py).
__device__ __forceinline__ int64_t events_total(const int64_t* total, int64_t capacity) {
  const int64_t t = *total;
  return t < capacity ? t : capacity;
}

__global__ void deliver_buf_kernel(
    const int64_t* __restrict__ synapses, const int64_t* __restrict__ pre_of,
    const int64_t* __restrict__ total, int64_t capacity,
    const int64_t* __restrict__ path_post, const int64_t* __restrict__ path_fanin,
    const float* __restrict__ net_weight, const int64_t* __restrict__ path_start, const uint8_t* __restrict__ path_flags, int64_t paths,
    float ach_suppression, const float* __restrict__ acetylcholine,
    const bool* __restrict__ dendrite_pre, const bool* __restrict__ dendrite_post,
    const int8_t* __restrict__ net_branch, int64_t branches,
    const bool* __restrict__ typed_pre, const int64_t* __restrict__ cell_type, int64_t kinds,
    const float* __restrict__ stp_u4, const float* __restrict__ stp_x4, const float* __restrict__ stp_U4,
    const bool* __restrict__ is_lamp5, const float* __restrict__ ngf_d, float lamp5_scale, int64_t lamp5_kind,
    const float* __restrict__ stp_x, const float* __restrict__ inh_gain,
    float* __restrict__ dendrite, float* __restrict__ drive, float* __restrict__ glutamate,
    float* __restrict__ inhibition, float* __restrict__ ngf) {
  const int64_t end = events_total(total, capacity);
  const float ach = ach_suppression > 0.0f ? *acetylcholine * ach_suppression : 0.0f;
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < end; e += (int64_t)gridDim.x * blockDim.x) {
    const int64_t syn = synapses[e];
    const int64_t pre = pre_of[e];
    // Признаки пути: бит 0 — на дендрит, бит 1 — подавляет ацетилхолин,
    // бит 2 — без истощения пресинапса.
    const int64_t path = path_of(path_start, paths, syn);
    const int64_t post = path_post[path] + (syn - path_start[path]) / path_fanin[path];
    float w = net_weight[syn];
    const uint8_t flags = path_flags[path];
    if (flags & 2) w *= 1.0f - ach;
    // Рабочая память -> смысл: на дальний дендрит (вес до истощения).
    if ((dendrite_pre[pre] && dendrite_post[post]) || (flags & 1)) {
      atomicAdd(dendrite + post * branches + net_branch[syn], w);
      continue;
    }
    const bool typed = typed_pre[pre];
    if (typed) {
      int64_t k = cell_type[post];
      const int64_t slot = pre * kinds + (k < 0 ? 0 : k);
      const float u = stp_u4[slot];
      float gain = u * stp_x4[slot] / fmaxf(1.0f - u, 1e-3f) / stp_U4[slot];
      if (is_lamp5[pre]) gain *= ngf_d[pre] * lamp5_scale;
      w *= gain;
    } else if (!(flags & 4)) {
      // Бит 2 — путь без истощения (другой выход клетки с депрессией пути).
      w *= stp_x[pre];
    }
    if (w < 0.0f) {
      w *= inh_gain[post];
      atomicAdd((cell_type[pre] == lamp5_kind ? ngf : inhibition) + post, w);
    } else if (w > 0.0f) {
      atomicAdd(drive + post, w);
      if (glutamate) atomicAdd(glutamate + post, w);
    }
  }
}

__global__ void ltd_buf_kernel(
    const int64_t* __restrict__ synapses, const int64_t* __restrict__ pre_of,
    const int64_t* __restrict__ total, int64_t capacity,
    const int64_t* __restrict__ path_start, const int64_t* __restrict__ path_post,
    const int64_t* __restrict__ path_fanin, int64_t paths, const float* __restrict__ trace_post, const float* __restrict__ ltd_mask,
    const float* __restrict__ symmetric_area, float a_plus, float a_minus,
    const float* __restrict__ net_weight, const float* __restrict__ ltd_ref,
    const float* __restrict__ scale, __half* __restrict__ eligibility) {
  const int64_t end = events_total(total, capacity);
  const float inv = 1.0f / *scale;
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < end; e += (int64_t)gridDim.x * blockDim.x) {
    const int64_t syn = synapses[e];
    const int64_t pre = pre_of[e];
    const int64_t path = path_of(path_start, paths, syn);
    const int64_t post = path_post[path] + (syn - path_start[path]) / path_fanin[path];
    float sign = -a_minus;
    if (symmetric_area) {
      // Симметричное окно: пре после поста — тоже потенциация.
      const float zone = symmetric_area[post];
      if (zone > 0.0f && zone == symmetric_area[pre]) sign = a_plus;
    }
    // LTD пропорциональна весу (Bi & Poo 1998; van Rossum et al. 2000):
    // ltd_ref пути — 1 / средний врождённый вес, 0 — без зависимости.
    if (ltd_ref && sign < 0.0f && ltd_ref[path] > 0.0f) sign *= net_weight[syn] * ltd_ref[path];
    const __half add = __float2half(trace_post[post] * ltd_mask[pre] * (sign * inv));
    eligibility[syn] = __hadd(eligibility[syn], add);  // синапс в списке шага один раз
  }
}

__global__ void ltp_buf_kernel(
    const int64_t* __restrict__ synapses, const int64_t* __restrict__ total, int64_t capacity,
    const int32_t* __restrict__ net_pre, const float* __restrict__ ltp_buf_kernel_v, float a_plus,
    const float* __restrict__ scale, __half* __restrict__ eligibility) {
  const int64_t end = events_total(total, capacity);
  const float inv = 1.0f / *scale;
  for (int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; e < end; e += (int64_t)gridDim.x * blockDim.x) {
    const int64_t syn = synapses[e];
    const __half add = __float2half(ltp_buf_kernel_v[net_pre[syn]] * (a_plus * inv));
    eligibility[syn] = __hadd(eligibility[syn], add);
  }
}


void deliver_buf(torch::Tensor synapses, torch::Tensor pre_of, torch::Tensor total,
             torch::Tensor path_post, torch::Tensor path_fanin, torch::Tensor net_weight,
             torch::Tensor path_start, torch::Tensor path_flags, bool ach_paths, double ach_suppression,
             torch::Tensor acetylcholine, torch::Tensor dendrite_pre, torch::Tensor dendrite_post,
             torch::Tensor net_branch, int64_t branches,
             torch::Tensor typed_pre, torch::Tensor cell_type, int64_t kinds,
             torch::Tensor stp_u4, torch::Tensor stp_x4, torch::Tensor stp_U4,
             torch::Tensor is_lamp5, torch::Tensor ngf_d, double lamp5_scale, int64_t lamp5_kind,
             torch::Tensor stp_x, torch::Tensor inh_gain,
             torch::Tensor dendrite, torch::Tensor drive, c10::optional<torch::Tensor> glutamate,
             torch::Tensor inhibition, torch::Tensor ngf) {
  auto stream = at::cuda::getCurrentCUDAStream();
  deliver_buf_kernel<<<BLOCKS, THREADS, 0, stream>>>(
      synapses.data_ptr<int64_t>(), pre_of.data_ptr<int64_t>(), total.data_ptr<int64_t>(), synapses.size(0),
      path_post.data_ptr<int64_t>(), path_fanin.data_ptr<int64_t>(), net_weight.data_ptr<float>(),
      path_start.data_ptr<int64_t>(), path_flags.data_ptr<uint8_t>(), path_start.size(0),
      ach_paths ? (float)ach_suppression : 0.0f, acetylcholine.data_ptr<float>(),
      dendrite_pre.data_ptr<bool>(), dendrite_post.data_ptr<bool>(),
      net_branch.data_ptr<int8_t>(), branches,
      typed_pre.data_ptr<bool>(), cell_type.data_ptr<int64_t>(), kinds,
      stp_u4.data_ptr<float>(), stp_x4.data_ptr<float>(), stp_U4.data_ptr<float>(),
      is_lamp5.data_ptr<bool>(), ngf_d.data_ptr<float>(), (float)lamp5_scale, lamp5_kind,
      stp_x.data_ptr<float>(), inh_gain.data_ptr<float>(),
      dendrite.data_ptr<float>(), drive.data_ptr<float>(),
      glutamate.has_value() ? glutamate->data_ptr<float>() : nullptr,
      inhibition.data_ptr<float>(), ngf.data_ptr<float>());
}

void ltd_buf(torch::Tensor synapses, torch::Tensor pre_of, torch::Tensor total,
         torch::Tensor path_start, torch::Tensor path_post, torch::Tensor path_fanin,
         torch::Tensor trace_post, torch::Tensor ltd_mask, c10::optional<torch::Tensor> symmetric_area,
         double a_plus, double a_minus, torch::Tensor net_weight, c10::optional<torch::Tensor> ltd_ref,
         torch::Tensor scale, torch::Tensor eligibility) {
  auto stream = at::cuda::getCurrentCUDAStream();
  ltd_buf_kernel<<<BLOCKS, THREADS, 0, stream>>>(
      synapses.data_ptr<int64_t>(), pre_of.data_ptr<int64_t>(), total.data_ptr<int64_t>(), synapses.size(0),
      path_start.data_ptr<int64_t>(), path_post.data_ptr<int64_t>(), path_fanin.data_ptr<int64_t>(),
      path_start.size(0), trace_post.data_ptr<float>(), ltd_mask.data_ptr<float>(),
      opt<float>(symmetric_area), (float)a_plus, (float)a_minus,
      net_weight.data_ptr<float>(), opt<float>(ltd_ref), scale.data_ptr<float>(),
      reinterpret_cast<__half*>(eligibility.data_ptr()));
}

void ltp_buf(torch::Tensor synapses, torch::Tensor total, torch::Tensor net_pre, torch::Tensor ltp_buf_kernel_v,
         double a_plus, torch::Tensor scale, torch::Tensor eligibility) {
  auto stream = at::cuda::getCurrentCUDAStream();
  ltp_buf_kernel<<<BLOCKS, THREADS, 0, stream>>>(
      synapses.data_ptr<int64_t>(), total.data_ptr<int64_t>(), synapses.size(0),
      net_pre.data_ptr<int32_t>(), ltp_buf_kernel_v.data_ptr<float>(), (float)a_plus, scale.data_ptr<float>(),
      reinterpret_cast<__half*>(eligibility.data_ptr()));
}

// Химическая пластичность всех пластичных синапсов разом (они первые в
// общем массиве): вес += след × шаг пути, в [0, потолок пути], свежая
// часть += изменение, след тратится у путей, где он сразу становится весом.
// Раньше — цикл Python по путям, 5–6 мелких операций на путь каждые 50 мс.
__global__ void plastic_update_kernel(
    float* __restrict__ weight, __half* __restrict__ eligibility, __half* __restrict__ early,
    const int64_t* __restrict__ path_start, int64_t paths, const float* __restrict__ step,
    const float* __restrict__ cap, const uint8_t* __restrict__ spend, int64_t total) {
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t path = path_of(path_start, paths, i);
    const float k = step[path];
    if (k == 0.0f) continue;
    const float before = weight[i];
    // Без слияния в FMA — тот же счёт, что у PyTorch (умножить, потом сложить).
    const float after = fminf(fmaxf(__fadd_rn(before, __fmul_rn(__half2float(eligibility[i]), k)), 0.0f), cap[path]);
    weight[i] = after;
    early[i] = __hadd(early[i], __float2half(after - before));
    if (spend[path]) eligibility[i] = __float2half(0.0f);
  }
}

void plastic_update(torch::Tensor weight, torch::Tensor eligibility, torch::Tensor early,
                    torch::Tensor path_start, torch::Tensor step, torch::Tensor cap, torch::Tensor spend,
                    int64_t total) {
  auto stream = at::cuda::getCurrentCUDAStream();
  plastic_update_kernel<<<BLOCKS, THREADS, 0, stream>>>(
      weight.data_ptr<float>(), reinterpret_cast<__half*>(eligibility.data_ptr()),
      reinterpret_cast<__half*>(early.data_ptr()), path_start.data_ptr<int64_t>(), path_start.size(0),
      step.data_ptr<float>(), cap.data_ptr<float>(), spend.data_ptr<uint8_t>(), total);
}

// Гетеросинаптическое сохранение суммы (Royer & Paré 2003; Chistiakova et
// al. 2014): усиление одних входов клетки ослабляет остальные, и наоборот —
// сумма входа строки пути [получатель × входы] возвращается к целевой.
// Блок на строку: сумма, потом все веса и свежая часть × цель / сумма.
__global__ void conserve_rows_kernel(
    float* __restrict__ weight, __half* __restrict__ early, const int64_t* __restrict__ row_start,
    const int32_t* __restrict__ row_len, const float* __restrict__ row_target, const float* __restrict__ row_cap) {
  const int64_t row = blockIdx.x;
  const int64_t base = row_start[row];
  const int32_t len = row_len[row];
  __shared__ float part[256];
  float sum = 0.0f;
  for (int32_t j = threadIdx.x; j < len; j += blockDim.x) sum += weight[base + j];
  part[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
    __syncthreads();
  }
  const float total = part[0];
  if (total <= 0.0f) return;
  const float factor = row_target[row] / total;
  const float cap = row_cap[row];
  for (int32_t j = threadIdx.x; j < len; j += blockDim.x) {
    const float before = weight[base + j];
    const float after = fminf(before * factor, cap);
    weight[base + j] = after;
    early[base + j] = __hadd(early[base + j], __float2half(after - before));
  }
}

void conserve_rows(torch::Tensor weight, torch::Tensor early, torch::Tensor row_start, torch::Tensor row_len,
                   torch::Tensor row_target, torch::Tensor row_cap) {
  const int64_t rows = row_start.size(0);
  if (rows == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  conserve_rows_kernel<<<rows, 256, 0, stream>>>(
      weight.data_ptr<float>(), reinterpret_cast<__half*>(early.data_ptr()), row_start.data_ptr<int64_t>(),
      row_len.data_ptr<int32_t>(), row_target.data_ptr<float>(), row_cap.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compact_events", &compact_events, "Плотный список событий шага");
  m.def("deliver_buf", &deliver_buf, "Прежняя доставка через буфер (сверка)");
  m.def("ltd_buf", &ltd_buf, "Прежняя LTD через буфер (сверка)");
  m.def("ltp_buf", &ltp_buf, "Прежняя LTP через буфер (сверка)");
  m.def("deliver", &deliver, "Доставка спайков шага");
  m.def("ltd", &ltd, "LTD по событиям шага");
  m.def("ltp", &ltp, "LTP по событиям шага");
  m.def("conserve_rows", &conserve_rows, "Сохранение суммы входа строки");
  m.def("plastic_update", &plastic_update, "Химическая пластичность всех путей");
}
