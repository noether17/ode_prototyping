#include "ode.hpp"

#include <vector>

#include <gtest/gtest.h>

TEST(ODETest, Test)
{
    EXPECT_EQ(1, 1);
}

TEST(EulerTest, EulerStepIncrementsOneDimensionalState)
{
    auto state = std::vector<double>{1.0};
    auto const dt = 0.1;
    auto const f = []() { return std::vector<double>{2.0}; };

    euler_step(std::span{state}, dt, f);

    EXPECT_DOUBLE_EQ(state[0], 1.2);
}

TEST(EulerTest, EulerStepIncrementsTwoDimensionalState)
{
    auto state = std::vector<double>{1.0, 2.0};
    auto const dt = 0.1;
    auto const f = []() { return std::vector<double>{2.0, 3.0}; };

    euler_step(std::span{state}, dt, f);

    EXPECT_DOUBLE_EQ(state[0], 1.2);
    EXPECT_DOUBLE_EQ(state[1], 2.3);
}