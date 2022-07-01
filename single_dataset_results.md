# Macro-F1 score for single datasets

|          |  ans | asap | nlpcc | \|  |  arc | argmin | fnc1 | iac1 | ibmcs | perspectrum | semeval2016t6 | snopes | twitter2015 | twitter2017 | vast |
|---------:|:----:|:----:|:-----:|:---:|:----:|:------:|:----:|:----:|:-----:|:-----------:|:-------------:|:------:|:-----------:|:-----------:|:----:|
|  w/o mlm | 0.87 | 0.20 |  0.52 | \|  | 0.31 |  0.51  | 0.97 | 0.40 |  0.71 |     0.83    |      0.23     |  0.75  |     0.65    |     0.67    | 0.49 |
| with mlm | 0.75 |      |  0.45 | \|  | 0.62 |        | 0.74 |      |       |             |               |        |             |             |      |

# Macro-F1 score for non-stance

| PAWSX-Zh | PAWSX-En | Id-Clickbait | Amazon-Zh |
|:--------:|:--------:|:------------:|:---------:|
|   0.85   |   0.93   |     0.79     |    0.54   |


# Hyperparameters
<table>
<tr><td>Learning Rate</td><td> 5e-6 </td></tr>
<tr><td>Number of Train Epochs</td><td> 20 </td></tr>
<tr><td>Logging / Save Steps</td><td> 500 </td></tr>
<tr><td>MLM alpha</td><td> 0.5 </td></tr>
</table>
