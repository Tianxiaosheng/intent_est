#pragma once

#include <string>

#include "intent_demo/types.h"

namespace intent_demo {

ObservationSequence LoadObservationsFromCsv(const std::string& path);
void SaveOutputsToCsv(const std::string& path, const EstimatorOutputs& outputs);

}  // namespace intent_demo
