# Mitigating Homophily Disparity in Graph Anomaly Detection: A Scalable and Adaptive Approach

Official Implementation of Mitigating Homophily Disparity in Graph Anomaly Detection: A Scalable and Adaptive Approach.



## Getting Started

### Setup Environment

To run the code, please install the following libraries: dgl==2.4.0+cu124, torch==2.4.0, numpy==2.1.3, scipy==1.14.1



### Preparing Datasets

We use [ten datasets](https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing) provided by [GADBench](https://github.com/squareRoot3/GADBench). After downloading, unzip all the files into the `datasets` folder.

Due to the Copyright of [DGraph-Fin](https://dgraph.xinye.com/introduction) and [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), you need to download these datasets by yourself. The script to preprocess DGraph-Fin and Elliptic can be found in [GADBench/preprocess.ipynb](https://github.com/squareRoot3/GADBench/blob/master/preprocess.ipynb). You can also preprocess your own dataset according to the notebook.



### Model Configuration

Model configurations/hyperparameters are provided in the `semi_train.conf.yaml`.



### Training and Evaluation

```bash
python main.py --dataset reddit
```



## Acknowledgements

The code is implemented based on [GADBench](https://github.com/squareRoot3/GADBench), [UniGAD](https://github.com/lllyyq1121/UniGAD), and [PolyGCL](https://github.com/ChenJY-Count/PolyGCL).