#!/usr/bin/env bash

set -e

python -m featuresharing.visualization.featurecontribution \
--datadir data/processed/Object+Text/marginalize/softmax \
--method rectangle \
--categories Object Text \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5 \
--sortby Text \
--log

python -m featuresharing.visualization.featurecontribution \
--datadir data/processed/Subord+Basic/marginalize/softmax \
--method rectangle \
--categories Subord Basic \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5 \
--sortby Basic \
--log
