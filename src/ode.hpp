#include <span>

// T is a floating point type
// AccFunc is a function object that returns a span of T
template <typename T, typename AccFunc>
void euler_step(std::span<T> pos, std::span<T> vel, T dt, AccFunc acc_func)
{
    auto const& acc = acc_func();
    for (std::size_t i = 0; i < pos.size(); ++i)
    {
        pos[i] += vel[i] * dt;
        vel[i] += acc[i] * dt;
    }
}