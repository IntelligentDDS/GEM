# GEM

This repo provides official code for "Subgraphs as First-Class Citizens in Incident Management for Large-Scale Online Systems: An Evolution-Aware Framework".

# Introduction

This work extends upon our previous publication ``Graph based Incident Extraction and Diagnosis in Large-Scale Online Systems'' at the 37th IEEE/ACM International Conference on Automated Software Engineering (ASE 2022)''.

## Project Structure

- `./data` contains the simulation environment dataset and open-sourced datasets used for helping the understanding and reporduction of each step of GEM.
- `./src` contains the implementation of GEM extracted for reproduction. 
- `./demo` contains ipython notebooks which provide examples to show how each step of GEM is performed. Their order is as follow:
    - `anomaly_detection_and_impact_extraction.ipynb` contains code for telemetry data anomaly detection and impact extraction.
    - `data_labelling.ipynb` contains code for data labelling using fault injection records.
    - `feature_engineering.ipynb` contains code for feature engineering.
    - `incident_detection.ipynb` contains code for the graph neural networks based model training and testing for incident detection on the simulation environment dataset.
    - `incident_diagnosis_using_edge_clues.ipynb` contains code for the incident diagnosis on the simulation environment dataset using edge clues.
    - `incident_diagnosis_using_node_clues_with_continual_optimization_OB.ipynb` contains code for the incident diagnosis on dataset OB using node clues with continual optimization.
    - `incident_diagnosis_using_node_clues_with_continual_optimization_AIOPS2021.ipynb` contains code for the incident diagnosis on dataset AIOPS2021 using node clues with continual optimization.

# Citation

If you find this work useful, please cite our paper:

```
@inproceedings{DBLP:conf/kbse/HeCLYCYL22,
  author       = {Zilong He and
                  Pengfei Chen and
                  Yu Luo and
                  Qiuyu Yan and
                  Hongyang Chen and
                  Guangba Yu and
                  Fangyuan Li},
  title        = {Graph based Incident Extraction and Diagnosis in Large-Scale Online
                  Systems},
  booktitle    = {37th {IEEE/ACM} International Conference on Automated Software Engineering,
                  {ASE} 2022, Rochester, MI, USA, October 10-14, 2022},
  pages        = {48:1--48:13},
  publisher    = {{ACM}},
  year         = {2022},
  url          = {https://doi.org/10.1145/3551349.3556904},
  doi          = {10.1145/3551349.3556904},
  timestamp    = {Thu, 22 Jun 2023 07:45:51 +0200},
  biburl       = {https://dblp.org/rec/conf/kbse/HeCLYCYL22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```



