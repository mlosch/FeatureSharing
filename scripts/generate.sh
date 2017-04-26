#!/usr/bin/env bash

set -e

# ----------------------------------------------------------------------------------------------------------------------
# Generate data and plots for Objects+Text task

python -m featuresharing.main \
--model data/models/alexnet-large-objectsvstextinthewild.t7 \
--images /media/max/Googolplex/Databases/MTL/raw/test/ \
--imagelist /home/max/Data/mtl_large/objects_vs_textinthewild/test.txt \
--categories Object Text \
--classtoidx data/models/meta.objectsvstext.t7 \
--output data/processed/alexnet-large-objectsvstextinthewild \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5 \
--softmax False \
--scoremethod "x"

python -m featuresharing.main \
--model data/models/alexnet-large-objectsvstextinthewild.t7 \
--images /media/max/Googolplex/Databases/MTL/raw/test/ \
--imagelist /home/max/Data/mtl_large/objects_vs_textinthewild/test.txt \
--categories Object Text \
--classtoidx data/models/meta.objectsvstext.t7 \
--output data/processed/alexnet-large-objectsvstextinthewild \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5 \
--softmax False \
--scoremethod "y"

python -m featuresharing.main \
--model data/models/alexnet-large-objectsvstextinthewild.t7 \
--images /media/max/Googolplex/Databases/MTL/raw/test/ \
--imagelist /home/max/Data/mtl_large/objects_vs_textinthewild/test.txt \
--categories Object Text \
--classtoidx data/models/meta.objectsvstext.t7 \
--output data/processed/alexnet-large-objectsvstextinthewild \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5 \
--softmax False \
--scoremethod "x" \
--random 20

# ----------------------------------------------------------------------------------------------------------------------
# Generate data and plots for Subordinate+Ordinate task

python -m featuresharing.main \
--model data/models/alexnet-large-basicvssubord.t7 \
--images /media/max/Googleplex/lost+found/#159318017/ \
--imagelist /home/max/Data/mtl_large/basic_vs_subord/test.txt \
--categories Subord Basic \
--classtoidx data/models/meta.basicvssubord.t7 \
--output data/processed/alexnet-large-basicvssubord \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5 \
--softmax False \
--scoremethod "x"

python -m featuresharing.main \
--model data/models/alexnet-large-basicvssubord.t7 \
--images /media/max/Googleplex/lost+found/#159318017/ \
--imagelist /home/max/Data/mtl_large/basic_vs_subord/test.txt \
--categories Subord Basic \
--classtoidx data/models/meta.basicvssubord.t7 \
--output data/processed/alexnet-large-basicvssubord \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5 \
--softmax False \
--scoremethod "y"

python -m featuresharing.main \
--model data/models/alexnet-large-basicvssubord.t7 \
--images /media/max/Googleplex/lost+found/#159318017/ \
--imagelist /home/max/Data/mtl_large/basic_vs_subord/test.txt \
--categories Subord Basic \
--classtoidx data/models/meta.basicvssubord.t7 \
--output data/processed/alexnet-large-basicvssubord \
--layers 0.0.0 0.0.3 0.0.6 0.0.8 0.0.10 1.2 1.5 \
--softmax False \
--scoremethod "x" \
--random 20