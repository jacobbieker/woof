// In-place launch of a translated CUDA kernel on MLX's Metal stream.
//
// mx.fast.metal_kernel allocates every output fresh, so a kernel that writes
// only some elements of a buffer leaves the rest undefined.  CUDA kernels
// mutate their pointer arguments in place, and woof's launchers depend on
// it.  LaunchTranslated is one MLX primitive that encodes a pipeline built
// from runtime MSL source, binds each pointer parameter at its byte offset,
// and makes every writable buffer's output share the input's storage
// (copy_shared_buffer), so unwritten words keep their values.
//
// Binding: a buffer goes to a Metal slot, or (past the 31-slot limit) into
// an argument table of GPU addresses; scalars arrive packed in blobs.
//
// Ordering: the output arrays depend on the inputs, so MLX schedules the
// launch after every producer of its buffers.  A lazy reader of an input's
// OLD value that is not an ancestor could run after the write;
// woof.backend.metal.runtime evaluates such readers before launching.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include <array>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/common/metal_kernel.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace nb = nanobind;
namespace mx = mlx::core;

// Pipelines built here, not by MLX's Device::get_kernel: MLX makes them from
// the bare function, which leaves MTLComputePipelineDescriptor's
// maxCallStackDepth at its default of 1.  A translated unit calls through
// several levels of functions the compiler keeps out of line (soft-float
// binary64, large device functions); deeper than the declared depth, Metal's
// results are undefined.  Breakage it prevents: Noah-MP's soil-water solve
// read __fma_rn(2, 3, 1) as 0 three calls deep and saturated every column.
constexpr NS::UInteger kCallStackDepth = 64;

static MTL::ComputePipelineState* pipeline_for(MTL::Library* lib, const std::string& library_key,
                                               const std::string& kernel) {
  static std::mutex mutex;
  static std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  std::string key = library_key + "\x1f" + kernel;
  auto it = cache.find(key);
  if (it != cache.end()) return it->second.get();
  auto pool = NS::AutoreleasePool::alloc()->init();
  auto* fn = lib->newFunction(NS::String::string(kernel.c_str(), NS::UTF8StringEncoding));
  if (fn == nullptr) {
    pool->release();
    throw std::runtime_error("[woof.backend.metal] no kernel " + kernel + " in its library");
  }
  auto* desc = MTL::ComputePipelineDescriptor::alloc()->init();
  desc->setComputeFunction(fn);
  desc->setMaxCallStackDepth(kCallStackDepth);
  NS::Error* error = nullptr;
  auto* state = mx::metal::device(mx::Device::gpu).mtl_device()->newComputePipelineState(
      desc, MTL::PipelineOptionNone, nullptr, &error);
  desc->release();
  fn->release();
  if (state == nullptr) {
    std::string what = error ? error->localizedDescription()->utf8String() : "unknown error";
    pool->release();
    throw std::runtime_error("[woof.backend.metal] cannot build the pipeline of " + kernel + ": " + what);
  }
  pool->release();
  cache.emplace(key, NS::TransferPtr(state));
  return state;
}

namespace {

using Dim3 = std::tuple<size_t, size_t, size_t>;

// Metal's last buffer slot: never bound by a translated kernel in table mode
// (table at 0, scalars at 1), used to show MLX the table's buffers.
constexpr int kHazardSlot = 30;

class LaunchTranslated : public mx::Primitive {
 public:
  LaunchTranslated(
      mx::Stream stream,
      std::string library_key,
      std::string source,
      std::string kernel,
      Dim3 groups,
      Dim3 threads,
      std::vector<int64_t> offsets,
      std::vector<bool> writable,
      std::vector<int> buffer_slots,
      int table_slot,
      std::vector<std::string> blobs,
      std::vector<int> blob_slots,
      size_t threadgroup_bytes,
      int math_mode,
      std::vector<int> null_slots,
      std::vector<int> null_table)
      : mx::Primitive(stream),
        library_key_(std::move(library_key)),
        source_(std::move(source)),
        kernel_(std::move(kernel)),
        groups_(groups),
        threads_(threads),
        offsets_(std::move(offsets)),
        writable_(std::move(writable)),
        buffer_slots_(std::move(buffer_slots)),
        table_slot_(table_slot),
        blobs_(std::move(blobs)),
        blob_slots_(std::move(blob_slots)),
        threadgroup_bytes_(threadgroup_bytes),
        math_mode_(math_mode),
        null_slots_(std::move(null_slots)),
        null_table_(std::move(null_table)) {}

  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&)
      override {
    throw std::runtime_error(
        "[woof.backend.metal] LaunchTranslated runs on the GPU stream only; "
        "the CPU backend launches through woof.backend.host");
  }

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override {
    auto& s = stream();
    auto& d = mx::metal::device(s.device);
    mx::CompileOptions options;
    options.math_mode = static_cast<mx::MathMode>(math_mode_);
    auto* lib = d.get_library(library_key_, options, [this]() { return source_; });
    auto* pipeline = pipeline_for(lib, library_key_, kernel_);
    size_t per_group = std::get<0>(threads_) * std::get<1>(threads_) * std::get<2>(threads_);
    if (per_group > pipeline->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "[woof.backend.metal] " + kernel_ + ": " + std::to_string(per_group) +
          " threads per threadgroup exceed the pipeline's limit of " +
          std::to_string(pipeline->maxTotalThreadsPerThreadgroup()));
    }
    auto& enc = mx::metal::get_command_encoder(s);
    // Write-after-read across encoders: MLX fences an encoder only after
    // earlier encoders whose OUTPUTS it reads (it never writes in place).
    // This kernel writes its buffers in place, and an earlier encoder -- in
    // this command buffer or one still in flight -- may still be reading
    // them.  So it runs in a pass of its own: the earlier work ends its pass
    // here, and this pass ends below with wait_in_flight, which waits on
    // every in-flight encoder's fence (CUDA's stream order, on the GPU).
    enc.end_encoding();
    enc.set_compute_pipeline_state(pipeline);

    // Every buffer must be dense memory of its own size: a broadcast or
    // strided view would hand the kernel a buffer holding fewer elements
    // than it indexes.
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i].flags().row_contiguous || inputs[i].data_size() != inputs[i].size()) {
        throw std::runtime_error(
            "[woof.backend.metal] buffer " + std::to_string(i) + " of " + kernel_ +
            " is not dense row-contiguous memory");
      }
    }
    // Writable buffers: the output IS the input's storage.
    std::vector<int> out_of(inputs.size(), -1);
    int o = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (writable_[i]) {
        outputs[o].copy_shared_buffer(inputs[i]);
        out_of[i] = o++;
      }
    }
    // Direct slots, or entries of the argument table (Metal binds at most 31
    // buffers; a table of GPU addresses reaches the rest).  Table entries
    // are resident through MLX's residency sets, and their writes are
    // registered so MLX orders later readers after this launch.
    std::vector<uint64_t> table;
    for (size_t i = 0; i < inputs.size(); ++i) {
      int slot = buffer_slots_[i];
      const mx::array& a = out_of[i] >= 0 ? outputs[out_of[i]] : inputs[i];
      if (slot >= 0) {
        if (out_of[i] >= 0) {
          enc.set_output_array(outputs[out_of[i]], slot, offsets_[i]);
        } else {
          enc.set_input_array(inputs[i], slot, offsets_[i]);
        }
      } else {
        auto* buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
        table.push_back(buf->gpuAddress() + uint64_t(a.offset()) + uint64_t(offsets_[i]));
        // MLX inserts barriers only for hazards it sees through bound
        // buffers.  A table entry is reached by address, so bind it at a
        // slot the kernel never reads (the last one): MLX then tracks its
        // reads and writes like any other buffer's.
        if (out_of[i] >= 0) {
          enc.set_output_array(outputs[out_of[i]], kHazardSlot);
        } else {
          enc.set_input_array(inputs[i], kHazardSlot);
        }
      }
    }
    // Null pointers: a direct slot left without a buffer reads as null in
    // the kernel; a table entry is the address 0, at its pointer's position.
    for (int slot : null_slots_) {
      enc.set_buffer(nullptr, slot);
    }
    for (int pos : null_table_) {   // ascending
      table.insert(table.begin() + std::min<size_t>(size_t(pos), table.size()), uint64_t(0));
    }
    if (table_slot_ >= 0) {
      if (table.empty()) table.push_back(0);
      enc.set_bytes(table.data(), int(table.size()), table_slot_);
    }
    for (size_t k = 0; k < blobs_.size(); ++k) {
      const std::string& bytes = blobs_[k];
      if (bytes.empty()) continue;
      enc.set_bytes(bytes.data(), int(bytes.size()), blob_slots_[k]);
    }
    if (threadgroup_bytes_ > 0) {
      // Metal requires threadgroup memory lengths in multiples of 16.
      enc.set_threadgroup_memory_length((threadgroup_bytes_ + 15) & ~size_t(15), 0);
    }
    // MLX encodes concurrently and fences only read-after-write.  An earlier
    // dispatch may still be reading a buffer this kernel writes in place.
    enc.barrier();
    auto [gx, gy, gz] = groups_;
    auto [tx, ty, tz] = threads_;
    enc.dispatch_threadgroups(MTL::Size(gx, gy, gz), MTL::Size(tx, ty, tz));
    enc.end_encoding(/* wait_in_flight = */ true);
  }

  const char* name() const override {
    return "LaunchTranslated";
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs)
      override {
    std::vector<mx::Shape> shapes;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (writable_[i]) shapes.push_back(inputs[i].shape());
    }
    return shapes;
  }

 private:
  std::string library_key_;
  std::string source_;
  std::string kernel_;
  Dim3 groups_;
  Dim3 threads_;
  std::vector<int64_t> offsets_;
  std::vector<bool> writable_;
  std::vector<int> buffer_slots_;
  int table_slot_;
  std::vector<std::string> blobs_;
  std::vector<int> blob_slots_;
  size_t threadgroup_bytes_;
  std::vector<int> null_slots_;
  std::vector<int> null_table_;
  int math_mode_;
};

// buffer_slots[i] >= 0 binds buffers[i] there; -1 puts its GPU address in
// the argument table (in buffer order) bound at table_slot.  blobs[k] is
// set_bytes'd at blob_slots[k].  Returns one array per writable buffer, in
// buffer order; each shares that buffer's storage.
std::vector<mx::array> launch(
    const std::string& library_key,
    const std::string& source,
    const std::string& kernel,
    Dim3 groups,
    Dim3 threads,
    const std::vector<mx::array>& buffers,
    const std::vector<int64_t>& offsets,
    const std::vector<bool>& writable,
    const std::vector<int>& buffer_slots,
    int table_slot,
    const std::vector<nb::bytes>& blobs,
    const std::vector<int>& blob_slots,
    size_t threadgroup_bytes,
    int math_mode,
    std::vector<int> null_slots,
    std::vector<int> null_table) {
  size_t n = buffers.size();
  if (offsets.size() != n || writable.size() != n || buffer_slots.size() != n) {
    throw std::invalid_argument("offsets, writable and buffer_slots must match buffers");
  }
  if (blobs.size() != blob_slots.size()) {
    throw std::invalid_argument("blobs and blob_slots must match");
  }
  if (math_mode < 0 || math_mode > 2) {
    throw std::invalid_argument("math_mode must be 0 (safe), 1 (relaxed) or 2 (fast)");
  }
  for (int slot : buffer_slots) {
    if (slot > 30) throw std::out_of_range("Metal buffer slots are 0..30");
  }
  std::vector<std::string> blob_bytes;
  for (const auto& b : blobs) blob_bytes.emplace_back(b.c_str(), b.size());
  std::vector<mx::Shape> shapes;
  std::vector<mx::Dtype> dtypes;
  for (size_t i = 0; i < n; ++i) {
    if (writable[i]) {
      shapes.push_back(buffers[i].shape());
      dtypes.push_back(buffers[i].dtype());
    }
  }
  if (shapes.empty()) {
    // MLX schedules a primitive only through its outputs.
    throw std::invalid_argument("kernel launch needs at least one writable buffer");
  }
  auto stream = mx::default_stream(mx::Device::gpu);
  auto primitive = std::make_shared<LaunchTranslated>(
      stream, library_key, source, kernel, groups, threads, offsets, writable,
      buffer_slots, table_slot, std::move(blob_bytes), blob_slots, threadgroup_bytes, math_mode,
      std::move(null_slots), std::move(null_table));
  return mx::array::make_arrays(std::move(shapes), dtypes, primitive, buffers);
}

// Build the library and each kernel's pipeline now, so a unit that does
// not compile fails where it is loaded rather than at a later evaluation.
// Returns each kernel's maxTotalThreadsPerThreadgroup.
std::vector<int> compile(
    const std::string& library_key,
    const std::string& source,
    const std::vector<std::string>& kernels,
    int math_mode) {
  auto& d = mx::metal::device(mx::Device::gpu);
  mx::CompileOptions options;
  options.math_mode = static_cast<mx::MathMode>(math_mode);
  auto* lib = d.get_library(library_key, options, [&]() { return source; });
  std::vector<int> limits;
  for (const auto& k : kernels) {
    limits.push_back(int(pipeline_for(lib, library_key, k)->maxTotalThreadsPerThreadgroup()));
  }
  return limits;
}

// The GPU address of an evaluated array's first element: what CUDA code
// stores when it builds tables of device pointers.  Valid while the array's
// buffer lives and stays resident (MLX's residency sets).
uint64_t gpu_address(const mx::array& a) {
  if (a.status() == mx::array::Status::unscheduled) {
    throw std::invalid_argument("gpu_address needs an evaluated array");
  }
  auto* buf = static_cast<const MTL::Buffer*>(a.buffer().ptr());
  return buf->gpuAddress() + uint64_t(a.offset());
}

}  // namespace

NB_MODULE(_woof_metal_launch, m) {
  m.doc() = "In-place launches of translated CUDA kernels on MLX's Metal stream.";
  m.def(
      "launch",
      &launch,
      nb::arg("library_key"),
      nb::arg("source"),
      nb::arg("kernel"),
      nb::arg("groups"),
      nb::arg("threads"),
      nb::arg("buffers"),
      nb::arg("offsets"),
      nb::arg("writable"),
      nb::arg("buffer_slots"),
      nb::arg("table_slot"),
      nb::arg("blobs"),
      nb::arg("blob_slots"),
      nb::arg("threadgroup_bytes") = 0,
      nb::arg("math_mode") = 0,
      nb::arg("null_slots") = std::vector<int>{},
      nb::arg("null_table") = std::vector<int>{});
  m.def("gpu_address", &gpu_address, nb::arg("array"));
  m.def("compile", &compile, nb::arg("library_key"), nb::arg("source"),
        nb::arg("kernels"), nb::arg("math_mode") = 0);
}
