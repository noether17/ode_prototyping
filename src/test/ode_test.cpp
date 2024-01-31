#include "ode.hpp"

#include <vector>

#include <gtest/gtest.h>

TEST(ODETest, Test)
{
    EXPECT_EQ(1, 1);
}

TEST(EulerTest, EulerStepImplWorksWithFirstOrderEquation)
{
    auto state = std::vector<double>{1.0, 2.0};
    auto const derivative = std::vector<double>{2.0, 3.0};
    auto const dt = 0.1;

    euler_step_impl(dt, std::forward_as_tuple(state, derivative), std::make_index_sequence<1>{});

    EXPECT_DOUBLE_EQ(state[0], 1.2);
    EXPECT_DOUBLE_EQ(state[1], 2.3);

}

TEST(EulerTest, EulerStepImplWorksWithSecondOrderEquation)
{
    auto pos = std::vector<double>{1.0, 2.0, 3.0};
    auto vel = std::vector<double>{4.0, 5.0, 6.0};
    auto const acc = std::vector<double>{7.0, 8.0, 9.0};
    auto const dt = 0.1;

    euler_step_impl(dt, std::forward_as_tuple(pos, vel, acc), std::make_index_sequence<2>{});

    EXPECT_DOUBLE_EQ(pos[0], 1.4);
    EXPECT_DOUBLE_EQ(pos[1], 2.5);
    EXPECT_DOUBLE_EQ(pos[2], 3.6);
    EXPECT_DOUBLE_EQ(vel[0], 4.7);
    EXPECT_DOUBLE_EQ(vel[1], 5.8);
    EXPECT_DOUBLE_EQ(vel[2], 6.9);
}

TEST(EulerTest, EulerStepIncrementsOneDimensionalState)
{
    auto state = std::vector<double>{1.0};
    auto const dt = 0.1;
    auto const f = []() { return std::vector<double>{2.0}; };

    euler_step(dt, state, f());

    EXPECT_DOUBLE_EQ(state[0], 1.2);
}

TEST(EulerTest, EulerStepIncrementsTwoDimensionalState)
{
    auto state = std::vector<double>{1.0, 2.0};
    auto const dt = 0.1;
    auto const f = []() { return std::vector<double>{2.0, 3.0}; };

    euler_step(dt, state, f());

    EXPECT_DOUBLE_EQ(state[0], 1.2);
    EXPECT_DOUBLE_EQ(state[1], 2.3);
}

TEST(EulerTest, EulerStepIncrementsSecondOrderEquation)
{
    auto pos = std::vector<double>{1.0, 2.0, 3.0};
    auto vel = std::vector<double>{4.0, 5.0, 6.0};
    auto const dt = 0.1;
    auto const f = []() { return std::vector<double>{7.0, 8.0, 9.0}; };

    euler_step(dt, pos, vel, f());

    EXPECT_DOUBLE_EQ(pos[0], 1.4);
    EXPECT_DOUBLE_EQ(pos[1], 2.5);
    EXPECT_DOUBLE_EQ(pos[2], 3.6);
    EXPECT_DOUBLE_EQ(vel[0], 4.7);
    EXPECT_DOUBLE_EQ(vel[1], 5.8);
    EXPECT_DOUBLE_EQ(vel[2], 6.9);
}