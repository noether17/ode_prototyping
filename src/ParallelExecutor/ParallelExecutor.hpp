#pragma once

#include <functional>
#include <latch>
#include <thread>
#include <vector>

template <typename Predicate>
bool spinlock(std::stop_token& stop_token, Predicate pred) {
  for (auto trial = 0; not stop_token.stop_requested(); ++trial) {
    if (pred()) {
      return true;
    }
    if (trial == 8) {
      trial = 0;
      std::this_thread::yield();
    }
  }
  return false;
}

class ParallelExecutor {
 public:
  explicit ParallelExecutor(int n_threads) : m_task_ready_flags(n_threads) {
    for (auto thread_id = 0; thread_id < n_threads; ++thread_id) {
      m_threads.emplace_back(
          [this, thread_id, n_threads](std::stop_token stop_token) {
            auto old_n_items = 0;
            auto thread_begin = 0;
            auto thread_end = 0;
            while (true) {
              if (not spinlock(stop_token, [this, thread_id] {
                    return m_task_ready_flags[thread_id].load();
                  })) {
                return;
              }
              m_task_ready_flags[thread_id] = false;

              if (auto current_n_items = m_n_items.load();
                  current_n_items != old_n_items) {
                old_n_items = current_n_items;
                auto items_per_thread =
                    (current_n_items + n_threads - 1) / n_threads;
                thread_begin = thread_id * items_per_thread;
                thread_end = std::min((thread_id + 1) * items_per_thread,
                                      current_n_items);
              }

              m_task(thread_begin, thread_end);
            }
          },
          std::stop_token{m_stop_source.get_token()});
    }
  }

  ~ParallelExecutor() { m_stop_source.request_stop(); }

  template <typename ParallelKernel, typename... Args>
  void call_parallel_kernel(ParallelKernel kernel, int n_items,
                            Args&&... args) {
    auto latch = std::latch{std::ssize(m_threads)};
    m_n_items = n_items;
    m_task = [&](int thread_begin, int thread_end) {
      for (auto i = thread_begin; i < thread_end; ++i) {
        kernel(i, args...);
      }
      latch.count_down();
    };
    for (auto& flag : m_task_ready_flags) {
      flag = true;
    }
    latch.wait();
  }

 private:
  std::stop_source m_stop_source{};
  std::function<void(int, int)> m_task{};
  std::vector<std::atomic_bool> m_task_ready_flags{};
  std::atomic_int m_n_items{};
  std::vector<std::jthread> m_threads{};
};
