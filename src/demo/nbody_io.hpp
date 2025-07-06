#include <fstream>
#include <string>

void output_to_file(std::string const& filename, auto const& output,
                    double softening = 0.0) {
  auto const n_times = static_cast<std::size_t>(output.times.size());
  auto const n_var = static_cast<std::size_t>(output.states[0].size());

  auto output_file = std::ofstream{filename, std::ios::out | std::ios::binary};
  output_file.write(reinterpret_cast<char const*>(&n_times), sizeof(n_times));
  output_file.write(reinterpret_cast<char const*>(&n_var), sizeof(n_var));
  output_file.write(reinterpret_cast<char const*>(&softening),
                    sizeof(softening));
  for (std::size_t i = 0; i < n_times; ++i) {
    output_file.write(reinterpret_cast<char const*>(&output.times[i]),
                      sizeof(output.times[i]));
    for (std::size_t j = 0; j < n_var; ++j) {
      output_file.write(reinterpret_cast<char const*>(&output.states[i][j]),
                        sizeof(output.states[i][j]));
    }
  }
}
