#!/usr/bin/env bash

set -e

python -m featuresharing.analysis.featurecontribution \
--datadir data/processed/alexnet-large-objectsvstextinthewild/Object+Text/marginalize/softmax \
--method we \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5

python -m featuresharing.analysis.featurecontribution \
--datadir data/processed/alexnet-large-basicvssubord/Subord+Basic/marginalize/softmax \
--method we \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5
