# FraudDetectionByCUSBoost
This repository contain dataset, codes and experiment result for the undergraduate thesis.

### Dataset Description

- Dataset: uscecchini28.csv

- The origin dataset is from https://github.com/JarFraud/FraudDetection

  > Yang Bao, Bin Ke, Bin Li, Julia Yu, and Jie Zhang (2020). [Detecting Accounting Fraud in Publicly Traded U.S. Firms Using a Machine Learning Approach](https://onlinelibrary.wiley.com/doi/10.1111/1475-679X.12292). Journal of Accounting Research, 58 (1): 199-235.

- This dataset contains 28 direct financial variables from the financial statement and a "misstate" mark for AAER database

### Code Description

- AdaBoost: AdaBoost.py
- RUSBoost: rus_sampling.py, rusboost.py
- CUSBoost: cus_sampling.py, cusboost.py
- run.py: run and evaluate the algorithm

### Experiment Result

- test.ipynb

### Document
- A mini paper (in English): mini_paper.pdf
