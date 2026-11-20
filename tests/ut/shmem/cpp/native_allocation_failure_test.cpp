/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>

#include "cann/host.h"
#include "runtime/log.h"
#include "runtime/runtime.h"

namespace {
bool fail_next_new = false;
bool fail_registration = false;
bool fail_physical_allocation = false;
int allocations = 0;
int frees = 0;
int finalizes = 0;

void check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

void *operator new(std::size_t bytes) {
  if (fail_next_new) {
    fail_next_new = false;
    throw std::bad_alloc();
  }
  if (void *address = std::malloc(bytes == 0 ? 1 : bytes)) return address;
  throw std::bad_alloc();
}
void operator delete(void *address) noexcept { std::free(address); }
void operator delete(void *address, std::size_t) noexcept { std::free(address); }

// Substitute the external backend, retaining the production Runtime and registry.
namespace hyper_multicore::shmem::cann::host {
runtime::Status initialize(const InitOptions &) { return {}; }
runtime::Status finalize() { ++finalizes; return {}; }
runtime::Result<int32_t> query_current_device_index() { return runtime::Result<int32_t>::Success(0); }
runtime::Result<DeviceIdentity> query_device_identity() {
  return runtime::Result<DeviceIdentity>::Success({0, kBackendNpuArch, "Ascend910B"});
}
runtime::Result<uintptr_t> allocate(uint64_t) {
  if (fail_physical_allocation) {
    return runtime::Result<uintptr_t>::Failure(
      {runtime::ErrorCode::CannError, 1, "backend allocation failed", std::nullopt});
  }
  ++allocations;
  fail_next_new = fail_registration;
  return runtime::Result<uintptr_t>::Success(0x10000);
}
runtime::Result<uintptr_t> aligned_allocate(uint64_t, uint64_t bytes) { return allocate(bytes); }
void free_memory(uintptr_t) { ++frees; }
void barrier_on_stream(const runtime::StreamView &) {}
}  // namespace hyper_multicore::shmem::cann::host

namespace hyper_multicore::shmem::runtime::log {
bool is_enabled(Level) noexcept { return false; }
void Line(Level, int32_t, const char *, int, std::string_view, std::string_view, const char *, ...) noexcept {}
void Failure(const char *, int, const DfxFailure &) noexcept {}
}  // namespace hyper_multicore::shmem::runtime::log

int main() {
  using namespace hyper_multicore::shmem::runtime;  // NOLINT(build/namespaces)
  try {
    Runtime &runtime = Runtime::Instance();
    Config config{4096, 120, DataEngine::Mte, "tcp://127.0.0.1:8662"};
    check(runtime.Initialize({0, 1}, config, "tcp://127.0.0.1:8662").error_code == ErrorCode::Ok,
          "initialization failed");
    check(!runtime.Allocate({0, 0}).ok() && allocations == 0, "invalid arguments reached backend");
    fail_physical_allocation = true;
    check(!runtime.Allocate({64, 0}).ok(), "backend failure was lost");
    check(runtime.DebugState().state == State::Ready, "backend failure incorrectly invalidated Runtime");
    fail_physical_allocation = false;
    const auto record = runtime.Allocate({64, 0}).value();
    check(runtime.Free({record.allocation_id, record.allocation_base, 64, 0}).error_code == ErrorCode::Ok,
          "normal allocation could not be freed");
    fail_registration = true;
    bool original_error = false;
    try {
      runtime.Allocate({64, 0});
    } catch (const std::bad_alloc &) {
      original_error = true;
    }
    check(original_error, "registration lost the original bad_alloc");
    const auto snapshot = runtime.DebugState();
    check(snapshot.state == State::AllocationFailed, "registration failure left Runtime reusable");
    check(snapshot.allocation_failure.has_value(), "registration failure context was lost");
    const auto failure = *snapshot.allocation_failure;
    check(failure.stage == AllocationFailureStage::Register && failure.allocation_base == 0x10000 &&
          failure.allocation_bytes == 64, "registration failure context is incorrect");
    check(!runtime.Allocate({64, 0}).ok(), "terminal Runtime accepted another allocation");
    check(runtime.Free({record.allocation_id, record.allocation_base, 64, 0}).error_code != ErrorCode::Ok,
          "terminal Runtime accepted free");
    check(runtime.Shutdown().error_code != ErrorCode::Ok, "terminal Runtime accepted shutdown");
    check(allocations == 2 && frees == 1 && finalizes == 0, "failure performed backend cleanup or reuse");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
