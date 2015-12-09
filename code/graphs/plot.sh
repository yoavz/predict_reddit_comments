#!/bin/bash

export OUTPUT_NAME="${1}.png"
export INPUT_NAME="${1}.run"
export RANGE="${2}"
gnuplot error_vs_reg.plg 
