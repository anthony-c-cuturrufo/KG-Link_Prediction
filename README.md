# KG-Link_Prediction
## Setup
  Create environment with ```environment_py36_trec1.yml```\
  Note: Upgrade DGL to newest version\
  Create folder ```notebooks/train```\
  Run ```python lib/format_transformer.py --dataset <dataset name>``` to assemble data for link prediction.\
  See ```python lib/format_transformer.py -h``` for more information.

## Link Prediction
  Run ```python link_prediction.py -d <dataset name from setup>```
  
## Demo
  Check out ```demo.ipynb``` for a much more intuitive demonstration of the link prediction pipeline. 
