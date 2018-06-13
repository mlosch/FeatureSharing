#!/usr/bin/env bash

MODELDST="data/models"
mkdir -p "$MODELDST"
wget -cO "$MODELDST/models.tar.bz2" "https://datasets.d2.mpi-inf.mpg.de/multitaskcnns/models.tar.bz2"
tar -xvjf "$MODELDST/models.tar.bz2" -C "$MODELDST/" && rm "$MODELDST/models.tar.bz2"

DATADST="data"
wget -cO "$DATADST/mtldataset_test.tar.gz" "https://datasets.d2.mpi-inf.mpg.de/multitaskcnns/mtldataset_test.tar.gz"
tar -xvzf "$DATADST/mtldataset_test.tar.gz" -C "$DATADST/" && rm "$DATADST/mtldataset_test.tar.gz"
