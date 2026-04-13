#pragma once

#include <array>
#include <string>
#include <vector>

namespace intent_demo {

enum class HiddenMode {
    Yield = 0,
    Go = 1,
    Hesitate = 2,
};

inline constexpr std::size_t kModeCount = 3;

inline std::string ModeToString(HiddenMode mode) {
    switch (mode) {
        case HiddenMode::Yield:
            return "yield";
        case HiddenMode::Go:
            return "go";
        case HiddenMode::Hesitate:
            return "hesitate";
    }
    return "unknown";
}

struct Observation {
    double timestamp_s = 0.0;
    double dt_s = 0.1;

    double ego_distance_to_conflict_m = 0.0;
    double ego_speed_mps = 0.0;
    double ego_acc_mps2 = 0.0;

    double obj_distance_to_conflict_m = 0.0;
    double obj_speed_mps = 0.0;
    double obj_acc_mps2 = 0.0;

    bool object_has_priority = false;
    bool object_has_yield_sign = false;
};

struct DerivedFeatures {
    double ego_time_to_conflict_s = 0.0;
    double obj_time_to_conflict_s = 0.0;
    double delta_time_to_conflict_s = 0.0;
    double delta_time_rate = 0.0;
    double ego_commit_score = 0.0;
    double obj_stop_proximity_score = 0.0;
    double obj_decel_score = 0.0;
    double obj_accel_score = 0.0;
};

struct ContinuousParameters {
    double accepted_gap_s = 1.5;
    double aggressiveness = 0.5;
    double response_delay_s = 0.4;
};

struct EstimatorOutput {
    double timestamp_s = 0.0;
    std::array<double, kModeCount> mode_probabilities{};
    ContinuousParameters parameters{};
    DerivedFeatures features{};
};

using ObservationSequence = std::vector<Observation>;
using EstimatorOutputs = std::vector<EstimatorOutput>;

}  // namespace intent_demo
