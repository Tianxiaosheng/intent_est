#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "intent_demo/csv_io.h"
#include "intent_demo/intent_estimator.h"
#include "intent_demo/types.h"

namespace {

void PrintUsage(const char* executable) {
    std::cout << "Usage: " << executable << " [input_csv] [output_csv]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string input_path = argc > 1 ? argv[1] : "data/sample_observations_2.csv";
        const std::string output_path = argc > 2 ? argv[2] : "output/sample_outputs.csv";

        if (argc > 3) {
            PrintUsage(argv[0]);
            return 1;
        }

        const auto observations = intent_demo::LoadObservationsFromCsv(input_path);
        if (observations.empty()) {
            throw std::runtime_error("No observations found in input csv.");
        }

        intent_demo::IntentEstimator estimator;
        intent_demo::EstimatorOutputs outputs;
        outputs.reserve(observations.size());

        for (const auto& observation : observations) {
            outputs.push_back(estimator.Update(observation));
        }

        intent_demo::SaveOutputsToCsv(output_path, outputs);

        std::cout << "Processed " << observations.size() << " frames.\n";
        std::cout << "Input : " << std::filesystem::absolute(input_path) << "\n";
        std::cout << "Output: " << std::filesystem::absolute(output_path) << "\n\n";

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "timestamp  p_yield  p_go    p_hes    gap_s   aggr   yld_dec\n";
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            if (index % 5 != 0 && index + 1 != outputs.size()) {
                continue;
            }
            const auto& row = outputs[index];
            std::cout << std::setw(8) << row.timestamp_s << "  "
                      << std::setw(7) << row.mode_probabilities[0] << "  "
                      << std::setw(6) << row.mode_probabilities[1] << "  "
                      << std::setw(7) << row.mode_probabilities[2] << "  "
                      << std::setw(6) << row.parameters.accepted_gap_s << "  "
                      << std::setw(6) << row.parameters.aggressiveness << "  "
                      << std::setw(8) << row.parameters.yield_deceleration_mps2 << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "intent_demo failed: " << error.what() << '\n';
        return 1;
    }
}
