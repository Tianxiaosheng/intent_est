#include "intent_demo/csv_io.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace intent_demo {
namespace {

std::vector<std::string> SplitCsvLine(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream stream(line);
    std::string item;
    while (std::getline(stream, item, ',')) {
        parts.push_back(item);
    }
    return parts;
}

bool ParseBool(const std::string& text) {
    return text == "1" || text == "true" || text == "TRUE" || text == "True";
}

}  // namespace

ObservationSequence LoadObservationsFromCsv(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open observation csv: " + path);
    }

    ObservationSequence observations;
    std::string line;
    bool is_header = true;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        if (is_header) {
            is_header = false;
            continue;
        }

        const auto columns = SplitCsvLine(line);
        if (columns.size() != 10) {
            throw std::runtime_error("Unexpected column count in csv line: " + line);
        }

        Observation observation{};
        observation.timestamp_s = std::stod(columns[0]);
        observation.dt_s = std::stod(columns[1]);
        observation.ego_distance_to_conflict_m = std::stod(columns[2]);
        observation.ego_speed_mps = std::stod(columns[3]);
        observation.ego_acc_mps2 = std::stod(columns[4]);
        observation.obj_distance_to_conflict_m = std::stod(columns[5]);
        observation.obj_speed_mps = std::stod(columns[6]);
        observation.obj_acc_mps2 = std::stod(columns[7]);
        observation.object_has_priority = ParseBool(columns[8]);
        observation.object_has_yield_sign = ParseBool(columns[9]);
        observations.push_back(observation);
    }

    return observations;
}

void SaveOutputsToCsv(const std::string& path, const EstimatorOutputs& outputs) {
    std::filesystem::path output_path(path);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream output(path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open output csv: " + path);
    }

    output << "timestamp_s,p_yield,p_go,p_hesitate,accepted_gap_s,aggressiveness,response_delay_s,delta_ttc_s,delta_ttc_rate,ego_commit_score,obj_stop_proximity_score\n";
    for (const auto& row : outputs) {
        output << row.timestamp_s << ','
               << row.mode_probabilities[0] << ','
               << row.mode_probabilities[1] << ','
               << row.mode_probabilities[2] << ','
               << row.parameters.accepted_gap_s << ','
               << row.parameters.aggressiveness << ','
               << row.parameters.response_delay_s << ','
               << row.features.delta_time_to_conflict_s << ','
               << row.features.delta_time_rate << ','
               << row.features.ego_commit_score << ','
               << row.features.obj_stop_proximity_score << '\n';
    }
}

}  // namespace intent_demo
