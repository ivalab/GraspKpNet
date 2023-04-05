The result of tuning hyper-parameters of alpha, beta and gamma on the Cornell Dataset is recorded in the following Table.
As shown in the Table, increasing weights of pushing, pulling and regression losses in the total loss function doesn't
improve the detection accuracy. The keypoint detection or focal loss should keep dominated.

Results of tuning hyper-parameters on the Cornell Dataset
| alpha:beta:gamma | 10:10:5 | 5:5:5 | 5:2.5:5 | 5:1:2.5 |
|:--------------------:|:----------:|:--------:|:--------:|:--------:|
|Accuracy (%) | 94.4 | 94.8 | 94.2 | 95.2 |
