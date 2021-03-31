# KG-Link_Prediction
## Setup
  Create environment with ```environment_py36_trec1.yml```\
  Create folder ```notebooks/train```\
  Run ```python lib/format_transformer.py --dataset <dataset name>``` to assemble data for link prediction.\
  See ```python lib/format_transformer.py -h``` for more information.

## Link Prediction
  Run ```python link_prediction.py -d <dataset name from setup>```
