#include "intent_demo/intent_estimator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace intent_demo {
namespace {

constexpr double kMinSpeedMps = 0.2;
constexpr double kTinyLikelihood = 1e-12;
constexpr double kSqrtTwoPi = 2.5066282746310002;

std::size_t ModeIndex(HiddenMode mode) {
    return static_cast<std::size_t>(mode);
}

}  // namespace

IntentEstimator::IntentEstimator(EstimatorConfig config)
    : config_(config), rng_(42U) {
    InitializeParticles();
}

void IntentEstimator::Reset() {
    initialized_ = false;
    previous_delta_time_to_conflict_s_ = 0.0;
    InitializeParticles();
}

EstimatorOutput IntentEstimator::Update(const Observation& observation) {
    if (particles_.empty()) {
        InitializeParticles();
    }

    DerivedFeatures features = BuildFeatures(observation);

    for (auto& particle : particles_) {
        const auto transition_probabilities =
            ComputeTransitionProbabilities(particle.mode, particle.parameters, observation, features);
        particle.mode = SampleMode(transition_probabilities);
        particle.parameters = PropagateParameters(particle.parameters);

        const double likelihood =
            ComputeObservationLikelihood(particle.mode, particle.parameters, observation, features);
        particle.weight *= std::max(likelihood, kTinyLikelihood);
    }

    NormalizeWeights();
    EstimatorOutput output = BuildOutput(observation, features);
    ResampleIfNeeded();

    previous_delta_time_to_conflict_s_ = features.delta_time_to_conflict_s;
    initialized_ = true;

    return output;
}

void IntentEstimator::InitializeParticles() {
    particles_.clear();
    particles_.resize(config_.particle_count);

    std::uniform_real_distribution<double> gap_distribution(0.9, 2.2);
    std::uniform_real_distribution<double> aggressiveness_distribution(0.2, 0.8);
    std::uniform_real_distribution<double> delay_distribution(0.2, 0.8);
    std::uniform_int_distribution<int> mode_distribution(0, static_cast<int>(kModeCount) - 1);

    const double initial_weight = 1.0 / static_cast<double>(config_.particle_count);
    for (auto& particle : particles_) {
        particle.mode = static_cast<HiddenMode>(mode_distribution(rng_));
        particle.parameters.accepted_gap_s = gap_distribution(rng_);
        particle.parameters.aggressiveness = aggressiveness_distribution(rng_);
        particle.parameters.response_delay_s = delay_distribution(rng_);
        particle.weight = initial_weight;
    }
}

DerivedFeatures IntentEstimator::BuildFeatures(const Observation& observation) const {
    DerivedFeatures features{};

    const double ego_speed = std::max(std::abs(observation.ego_speed_mps), kMinSpeedMps);
    const double obj_speed = std::max(std::abs(observation.obj_speed_mps), kMinSpeedMps);

    features.ego_time_to_conflict_s = observation.ego_distance_to_conflict_m / ego_speed;
    features.obj_time_to_conflict_s = observation.obj_distance_to_conflict_m / obj_speed;
    features.delta_time_to_conflict_s = features.obj_time_to_conflict_s - features.ego_time_to_conflict_s;

    const double dt = observation.dt_s > 1e-3 ? observation.dt_s : 0.1;
    features.delta_time_rate = initialized_
        ? (features.delta_time_to_conflict_s - previous_delta_time_to_conflict_s_) / dt
        : 0.0;

    const double commit_distance_score =
        1.0 - Clamp(observation.ego_distance_to_conflict_m / config_.ego_commit_distance_m, 0.0, 1.0);
    const double commit_speed_score = Clamp(observation.ego_speed_mps / config_.ego_commit_speed_mps, 0.0, 1.0);
    features.ego_commit_score = Clamp(0.55 * commit_distance_score + 0.45 * commit_speed_score, 0.0, 1.0);

    features.obj_stop_proximity_score =
        1.0 - Clamp(observation.obj_distance_to_conflict_m / config_.near_stop_distance_m, 0.0, 1.0);
    features.obj_decel_score = Clamp(-observation.obj_acc_mps2 / 3.0, 0.0, 1.0);
    features.obj_accel_score = Clamp(observation.obj_acc_mps2 / 2.5, 0.0, 1.0);

    return features;
}

std::array<double, kModeCount> IntentEstimator::ComputeTransitionProbabilities(
    HiddenMode previous_mode,
    const ContinuousParameters& parameters,
    const Observation& observation,
    const DerivedFeatures& features) const {
    static constexpr std::array<std::array<double, kModeCount>, kModeCount> kBaseTransitionBias = {{
        {{1.2, -0.3, 0.2}},
        {{-0.3, 1.2, 0.2}},
        {{0.1, 0.1, 0.8}},
    }};

    const double gap_margin = features.delta_time_to_conflict_s - parameters.accepted_gap_s;
    const double hesitation_window = 1.0 - Clamp(std::abs(gap_margin) / 1.5, 0.0, 1.0);
    const auto& row = kBaseTransitionBias[ModeIndex(previous_mode)];

    double score_yield = row[ModeIndex(HiddenMode::Yield)]
        + 1.40 * gap_margin
        - 1.10 * parameters.aggressiveness
        + 0.90 * features.obj_decel_score
        + 0.70 * features.ego_commit_score
        + (observation.object_has_yield_sign ? 0.40 : 0.0)
        - (observation.object_has_priority ? 0.80 : 0.0);

    double score_go = row[ModeIndex(HiddenMode::Go)]
        - 1.30 * gap_margin
        + 1.10 * parameters.aggressiveness
        + 0.80 * features.obj_accel_score
        + (observation.object_has_priority ? 0.70 : 0.0)
        - (observation.object_has_yield_sign ? 0.45 : 0.0)
        - 0.40 * features.ego_commit_score;

    double score_hesitate = row[ModeIndex(HiddenMode::Hesitate)]
        + 1.10 * hesitation_window
        + 0.35 * parameters.response_delay_s
        - 0.25 * std::abs(observation.obj_acc_mps2)
        + 0.15 * (1.0 - std::abs(features.delta_time_rate));

    const double max_score = std::max({score_yield, score_go, score_hesitate});
    score_yield = std::exp(score_yield - max_score);
    score_go = std::exp(score_go - max_score);
    score_hesitate = std::exp(score_hesitate - max_score);

    const double sum = score_yield + score_go + score_hesitate;
    return {
        score_yield / sum,
        score_go / sum,
        score_hesitate / sum,
    };
}

HiddenMode IntentEstimator::SampleMode(const std::array<double, kModeCount>& probabilities) {
    std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
    return static_cast<HiddenMode>(distribution(rng_));
}

ContinuousParameters IntentEstimator::PropagateParameters(const ContinuousParameters& parameters) {
    std::normal_distribution<double> gap_noise(0.0, config_.process_noise_gap);
    std::normal_distribution<double> aggressiveness_noise(0.0, config_.process_noise_aggressiveness);
    std::normal_distribution<double> delay_noise(0.0, config_.process_noise_delay);

    ContinuousParameters propagated = parameters;
    propagated.accepted_gap_s = Clamp(parameters.accepted_gap_s + gap_noise(rng_),
                                      config_.gap_min_s,
                                      config_.gap_max_s);
    propagated.aggressiveness = Clamp(parameters.aggressiveness + aggressiveness_noise(rng_),
                                      config_.aggressiveness_min,
                                      config_.aggressiveness_max);
    propagated.response_delay_s = Clamp(parameters.response_delay_s + delay_noise(rng_),
                                        config_.delay_min_s,
                                        config_.delay_max_s);
    return propagated;
}

double IntentEstimator::ComputeObservationLikelihood(HiddenMode mode,
                                                     const ContinuousParameters& parameters,
                                                     const Observation& observation,
                                                     const DerivedFeatures& features) const {
    const double expected_acc = ExpectedAcceleration(mode, parameters, observation, features);
    const double expected_delta_rate = ExpectedDeltaTimeRate(mode, parameters, observation, features);

    double likelihood = GaussianPdf(observation.obj_acc_mps2 - expected_acc, config_.sigma_acc) *
                        GaussianPdf(features.delta_time_rate - expected_delta_rate, config_.sigma_delta_rate);

    const double gap_margin = features.delta_time_to_conflict_s - parameters.accepted_gap_s;
    if (mode == HiddenMode::Yield) {
        likelihood *= 1.0 + 0.25 * Clamp((gap_margin + 0.5) / 2.5, 0.0, 1.0);
        if (observation.object_has_yield_sign) {
            likelihood *= 1.10;
        }
    } else if (mode == HiddenMode::Go) {
        likelihood *= 1.0 + 0.25 * Clamp((parameters.accepted_gap_s - gap_margin) / 2.5, 0.0, 1.0);
        if (observation.object_has_priority) {
            likelihood *= 1.10;
        }
    } else {
        likelihood *= 1.0 + 0.15 * (1.0 - Clamp(std::abs(gap_margin) / 1.5, 0.0, 1.0));
    }

    return std::max(likelihood, kTinyLikelihood);
}

double IntentEstimator::ExpectedAcceleration(HiddenMode mode,
                                             const ContinuousParameters& parameters,
                                             const Observation& observation,
                                             const DerivedFeatures& features) const {
    const double delta = features.delta_time_to_conflict_s;

    if (mode == HiddenMode::Yield) {
        const double brake_need = Clamp((parameters.accepted_gap_s - delta + 0.25) /
                                            std::max(parameters.accepted_gap_s + 0.25, 0.6),
                                        0.0,
                                        2.0);
        double expected = -0.15 - 1.80 * brake_need
            - 0.80 * features.ego_commit_score * (1.0 - 0.30 * parameters.aggressiveness)
            - 0.35 * features.obj_stop_proximity_score;
        if (observation.object_has_priority) {
            expected += 0.45;
        }
        if (observation.object_has_yield_sign) {
            expected -= 0.35;
        }
        return Clamp(expected, -4.0, 1.0);
    }

    if (mode == HiddenMode::Go) {
        const double push_need = Clamp((delta + 0.50) / (parameters.accepted_gap_s + 0.50), 0.0, 2.0);
        double expected = 0.10 + 1.60 * parameters.aggressiveness * (0.50 + push_need)
            + (observation.object_has_priority ? 0.60 : 0.0)
            - (observation.object_has_yield_sign ? 0.40 : 0.0)
            - 0.30 * features.obj_stop_proximity_score;
        return Clamp(expected, -1.0, 3.5);
    }

    const double gap_margin = delta - parameters.accepted_gap_s;
    const double neutralization = 1.0 - Clamp(std::abs(gap_margin) / 1.5, 0.0, 1.0);
    double expected = 0.25 * (parameters.aggressiveness - 0.5)
        - 0.45 * features.obj_stop_proximity_score * neutralization
        - 0.20 * features.ego_commit_score * (0.6 - parameters.aggressiveness);
    if (observation.object_has_priority) {
        expected += 0.10;
    }
    return Clamp(expected, -1.5, 1.5);
}

double IntentEstimator::ExpectedDeltaTimeRate(HiddenMode mode,
                                              const ContinuousParameters& parameters,
                                              const Observation& observation,
                                              const DerivedFeatures& features) const {
    if (mode == HiddenMode::Yield) {
        return 0.10
            + 0.20 * (1.0 - parameters.aggressiveness)
            + 0.18 * features.ego_commit_score
            + 0.08 * features.obj_decel_score
            - (observation.object_has_priority ? 0.06 : 0.0);
    }

    if (mode == HiddenMode::Go) {
        return -(0.10
            + 0.24 * parameters.aggressiveness
            + 0.12 * features.obj_accel_score
            + (observation.object_has_priority ? 0.10 : 0.0));
    }

    const double gap_margin = features.delta_time_to_conflict_s - parameters.accepted_gap_s;
    return -0.05 * Clamp(gap_margin, -1.0, 1.0) + 0.03 * (0.5 - parameters.aggressiveness);
}

void IntentEstimator::NormalizeWeights() {
    double sum = 0.0;
    for (const auto& particle : particles_) {
        sum += particle.weight;
    }

    if (sum <= std::numeric_limits<double>::epsilon()) {
        const double uniform_weight = 1.0 / static_cast<double>(particles_.size());
        for (auto& particle : particles_) {
            particle.weight = uniform_weight;
        }
        return;
    }

    for (auto& particle : particles_) {
        particle.weight /= sum;
    }
}

void IntentEstimator::ResampleIfNeeded() {
    double squared_weight_sum = 0.0;
    for (const auto& particle : particles_) {
        squared_weight_sum += particle.weight * particle.weight;
    }

    const double effective_sample_size = 1.0 / squared_weight_sum;
    if (effective_sample_size >= 0.5 * static_cast<double>(particles_.size())) {
        return;
    }

    std::vector<Particle> resampled;
    resampled.reserve(particles_.size());

    std::uniform_real_distribution<double> start_distribution(0.0, 1.0 / particles_.size());
    double cumulative_weight = particles_.front().weight;
    std::size_t index = 0;
    double threshold = start_distribution(rng_);

    for (std::size_t i = 0; i < particles_.size(); ++i) {
        const double target = threshold + static_cast<double>(i) / particles_.size();
        while (target > cumulative_weight && index + 1 < particles_.size()) {
            ++index;
            cumulative_weight += particles_[index].weight;
        }
        resampled.push_back(particles_[index]);
        resampled.back().weight = 1.0 / static_cast<double>(particles_.size());
    }

    particles_.swap(resampled);
}

EstimatorOutput IntentEstimator::BuildOutput(const Observation& observation,
                                             const DerivedFeatures& features) const {
    EstimatorOutput output{};
    output.timestamp_s = observation.timestamp_s;
    output.features = features;
    output.parameters = ContinuousParameters{0.0, 0.0, 0.0};

    for (const auto& particle : particles_) {
        output.mode_probabilities[ModeIndex(particle.mode)] += particle.weight;
        output.parameters.accepted_gap_s += particle.weight * particle.parameters.accepted_gap_s;
        output.parameters.aggressiveness += particle.weight * particle.parameters.aggressiveness;
        output.parameters.response_delay_s += particle.weight * particle.parameters.response_delay_s;
    }

    return output;
}

double IntentEstimator::Clamp(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

double IntentEstimator::Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double IntentEstimator::GaussianPdf(double residual, double sigma) {
    const double safe_sigma = std::max(sigma, 1e-6);
    const double normalized = residual / safe_sigma;
    return std::exp(-0.5 * normalized * normalized) / (safe_sigma * kSqrtTwoPi);
}

}  // namespace intent_demo
