// SPDX-License-Identifier: Apache-2.0
// MICM CPU oracle harness: musica v0.17.2 (MICM e15898c), RosenbrockStandardOrder, one grid cell per solve.
#include <musica/micm/micm.hpp>
#include <musica/micm/state.hpp>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: harness input\n"); return 2; }
  std::ifstream in(argv[1]);
  std::string key, config;
  double dt = 0;
  in >> key >> config; in >> key >> dt;
  musica::MICM micm(config, musica::MICMSolver::RosenbrockStandardOrder);
  std::map<std::string, std::size_t> order;
  for (auto& [n, i] : micm.GetSpeciesOrdering()) order[n] = i;
  std::map<std::string, std::size_t> rorder;
  for (auto& [n, i] : micm.GetRateParameterOrdering()) rorder[n] = i;
  for (auto& [n, i] : order) std::printf("order %s %zu\n", n.c_str(), i);
  for (auto& [n, i] : rorder) std::printf("rate_order %s %zu\n", n.c_str(), i);
  int ncase = 0;
  while (in >> key) {
    if (key != "case") break;
    musica::State state(micm, 1);
    micm::Conditions c{};
    std::map<std::string, std::vector<double>> conc, rates;
    while (in >> key && key != "end") {
      std::string name; double v;
      if (key == "T") { in >> c.temperature_; }
      else if (key == "P") { in >> c.pressure_; }
      else if (key == "rho") { in >> c.air_density_; }
      else if (key == "conc") { in >> name >> v; conc[name] = {v}; }
      else if (key == "rate") { in >> name >> v; rates[name] = {v}; }
    }
    state.SetConditions({c});
    state.SetConcentrations(conc, musica::MICMSolver::RosenbrockStandardOrder);
    if (!rates.empty()) state.SetRateConstants(rates, musica::MICMSolver::RosenbrockStandardOrder);
    auto result = micm.Solve(&state, dt);
    auto out = state.GetConcentrations(musica::MICMSolver::RosenbrockStandardOrder);
    std::printf("case %d state %d steps %llu accepted %llu rejected %llu\n", ncase, (int)result.state_,
                (unsigned long long)result.stats_.number_of_steps_, (unsigned long long)result.stats_.accepted_,
                (unsigned long long)result.stats_.rejected_);
    for (auto& [n, v] : out) std::printf("conc %s %a\n", n.c_str(), v[0]);
    ++ncase;
  }
  return 0;
}
