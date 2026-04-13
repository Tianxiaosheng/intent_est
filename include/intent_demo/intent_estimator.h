#pragma once

#include <random>
#include <vector>

#include "intent_demo/types.h"

namespace intent_demo {

struct EstimatorConfig {
    std::size_t particle_count = 300;

    double gap_min_s = 0.6;
    double gap_max_s = 3.0;
    double aggressiveness_min = 0.0;
    double aggressiveness_max = 1.0;
    double delay_min_s = 0.1;
    double delay_max_s = 1.5;

    double process_noise_gap = 0.05;
    double process_noise_aggressiveness = 0.03;
    double process_noise_delay = 0.02;

    double sigma_acc = 0.60;
    double sigma_delta_rate = 0.25;

    double ego_commit_distance_m = 8.0;
    double ego_commit_speed_mps = 1.5;
    double near_stop_distance_m = 10.0;
};

class IntentEstimator {
public:
    explicit IntentEstimator(EstimatorConfig config = {});

    void Reset();
    EstimatorOutput Update(const Observation& observation);

private:
    struct Particle {
        HiddenMode mode = HiddenMode::Hesitate;
        ContinuousParameters parameters{};
        double weight = 0.0;
    };

    EstimatorConfig config_{};
    std::vector<Particle> particles_;
    std::mt19937 rng_;
    bool initialized_ = false;
    double previous_delta_time_to_conflict_s_ = 0.0;

    void InitializeParticles();
    DerivedFeatures BuildFeatures(const Observation& observation) const;
    std::array<double, kModeCount> ComputeTransitionProbabilities(HiddenMode previous_mode,
                                                                  const ContinuousParameters& parameters,
                                                                  const Observation& observation,
                                                                  const DerivedFeatures& features) const;
    HiddenMode SampleMode(const std::array<double, kModeCount>& probabilities);
    ContinuousParameters PropagateParameters(const ContinuousParameters& parameters);
    double ComputeObservationLikelihood(HiddenMode mode,
                                        const ContinuousParameters& parameters,
                                        const Observation& observation,
                                        const DerivedFeatures& features) const;
    double ExpectedAcceleration(HiddenMode mode,
                                const ContinuousParameters& parameters,
                                const Observation& observation,
                                const DerivedFeatures& features) const;
    double ExpectedDeltaTimeRate(HiddenMode mode,
                                 const ContinuousParameters& parameters,
                                 const Observation& observation,
                                 const DerivedFeatures& features) const;
    void NormalizeWeights();
    void ResampleIfNeeded();
    EstimatorOutput BuildOutput(const Observation& observation, const DerivedFeatures& features) const;

    static double Clamp(double value, double lower, double upper);
    static double Sigmoid(double x);
    static double GaussianPdf(double residual, double sigma);
};

}  // namespace intent_demo
