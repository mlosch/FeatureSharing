#!/usr/bin/env bash

set -e

python -m featuresharing.preprocessing \
--model data/models/alexnet-large-objectsvstextinthewild.t7 \
--images data/images/ \
--imagelist data/labels/Object+Text/test.txt \
--categories Object Text \
--classtoidx data/models/meta.objectsvstext.t7 \
--output data/processed/alexnet-large-objectsvstextinthewild \
--method marginalize \
--samples 100 \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5

python -m featuresharing.preprocessing \
--model data/models/alexnet-large-basicvssubord.t7 \
--images data/images/ \
--imagelist data/labels/Basic+Subord/test.txt \
--categories Subord Basic \
--classtoidx data/models/meta.basicvssubord.t7 \
--output data/processed/alexnet-large-basicvssubord \
--method marginalize \
--samples 100 \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5
