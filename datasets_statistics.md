# Datasets

## English

### ARC
ARC splits are found from the mdl repo and topics are shared

|       | Against | Discussing | In Favor | Unrelated To | Total |
|-------|:-------:|:----------:|:--------:|:------------:|:-----:|
| Train |   1220  |     786    |   1094   |     9282     | 12382 |
| Dev   |   182   |     118    |    163   |     1388     | 1851  |
| Test  |   372   |     179    |    334   |     2674     | 3559  |
| Total |   1774  |    1083    |   1591   |    13344     | 17792 |

### ARGMIN
ArgMin are found from the mdl repo and 5 topics are used for training, 1 topic for dev and 2 topics for test

|       | Argument Against | Discussing | Argument For | Unrelated To | Total |
|-------|:----------------:|:----------:|:------------:|:------------:|:-----:|
| Train |       3690       |     0      |     3155     |      0       |  6845 |
| Dev   |       1111       |     0      |      457     |      0       |  1568 |
| Test  |       1394       |     0      |     1332     |      0       |  2726 |
| Total |       6195       |     0      |     4944     |              | 11139 |

### IAC1
IAC1 splits are found from the mdl repo w/o overlap in topics between the 3 sets
|       | Anti | Other |  Pro | Unrelated To | Total |
|-------|:----:|:-----:|:----:|:------------:|:-----:|
| Train | 1432 |  371  | 2424 |      0       |  4227 |
| Dev   |  117 |  113  |  224 |      0       |  454  |
| Test  |  395 |   96  |  433 |      0       |  924  |
| Total | 1944 |  580  | 3081 |      0       |  5605 |

### IBMCS
IBMCS splits are found from the mdl repo

|       |  Con | Discussing |  Pro | Unrelated To | Total |
|-------|:----:|:----------:|:----:|:------------:|:-----:|
| Train |  373 |     0      |  562 |      0       |  935  |
| Dev   |  41  |     0      |  63  |      0       |  104  |
| Test  |  655 |     0      |  700 |      0       |  1355 |
| Total | 1069 |     0      | 1325 |      0       |  2394 |

### FNC1
FNC1 splits are found from the mdl repo

|       | Disagree With | Discussing | Agree With | Unrelated To | Total |
|-------|:-------------:|:----------:|:----------:|:------------:|:-----:|
| Train |      714      |    7573    |    3126    |     31063    | 42476 |
| Dev   |      126      |    1336    |     552    |     5482     |  7496 |
| Test  |      697      |    4464    |    1903    |     18349    | 25413 |
| Total |     1537      |   13373    |    5581    |     54894    | 75385 |

### Perspectrum
Perspectrum splits are found from the mdl repo and topics are shared

|       | Undermines | Discussing | Supports | Unrelated To | Total |
|-------|:----------:|:----------:|:--------:|:------------:|:-----:|
| Train |    3379    |      0     |   3599   |       0      |  6978 |
| Dev   |    1024    |      0     |   1047   |       0      |  2071 |
| Test  |    1302    |      0     |   1471   |       0      |  2773 |
| Total |    5705    |      0     |   6117   |       0      | 11822 |

### SCD
SCD splits are found from the mdl repo w/o overlap in topics between the 3 sets

|       | Against |  For | Total |
|-------|:-------:|:----:|:-----:|
| Train |   1326  | 1925 |  3251 |
| Dev   |   181   |  443 |  624  |
| Test  |   438   |  526 |  964  |
| Total |   1945  | 2894 |  4839 |

### Semeval 2016 task 6
Semeval 2016 task 6 splits are found from the mdl repo and topics are shared

|       | Against | Discussing | In Favor | Unrelated To | Total |
|-------|:-------:|:----------:|:--------:|:------------:|:-----:|
| Train |   1191  |     657    |    649   |       0      |  2497 |
| Dev   |   204   |     109    |    104   |       0      |  417  |
| Test  |   715   |     230    |    304   |       0      |  1249 |
| Total |   2110  |     996    |   1057   |       0      |  4163 |

### Semeval 2019 task 7
Semeval 2019 task 7 splits are found from the mdl repo and there are no topics

|       | Support | Deny | Query | Comment | Total |
|-------|:-------:|:----:|:-----:|:-------:|:-----:|
| Train |   925   |  378 |  395  |   3519  |  5217 |
| Dev   |   102   |  82  |  120  |   1181  |  1485 |
| Test  |   157   |  101 |   93  |   1476  |  1827 |
| Total |   1184  |  561 |  608  |   6176  |  8529 |

### Snopes
Snopes splits are found from the mdl repo

|       | Refutes | Discussing | Agree With | Unrelated To | Total |
|-------|:-------:|:----------:|:----------:|:------------:|:-----:|
| Train |  3899   |     0      |   10517    |      0       | 14416 |
| Dev   |   515   |     0      |    1353    |      0       |  1868 |
| Test  |   678   |     0      |    2476    |      0       |  3154 |
| Total |  5092   |     0      |   14346    |      0       | 19438 |

### Twitter 2015
Twitter 2015 data are found from [TomBERT](https://github.com/jefferyYu/TomBERT)

|       | Negative | Neutral | Positve | Unrelated To | Total |
|-------|:--------:|:-------:|:-------:|:------------:|:-----:|
| Train |    368   |   1883  |   928   |      0       |  3179 |
| Dev   |    149   |   670   |   303   |      0       |  1122 |
| Test  |    113   |   607   |   317   |      0       |  1037 |
| Total |    630   |   3160  |   1548  |      0       |  5338 |

### Twitter 2017
Twitter 2017 data are found from [TomBERT](https://github.com/jefferyYu/TomBERT)

|       | Negative | Neutral | Positve | Unrelated To | Total |
|-------|:--------:|:-------:|:-------:|:------------:|:-----:|
| Train |    416   |   1638  |   1508  |      0       |  3562 |
| Dev   |    144   |   517   |   515   |      0       |  1176 |
| Test  |    168   |   573   |   493   |      0       |  1234 |
| Total |    728   |   2728  |   2516  |      0       |  5972 |

### VAST
Vast splits are found from [zero-shot-stance](https://github.com/emilyallaway/zero-shot-stance)

|       |  Con | Neutral |  Pro | Unrelated To | Total |
|-------|:----:|:-------:|:----:|:------------:|:-----:|
| Train | 5595 |   2555  | 5327 |      0       | 13477 |
| Dev   |  684 |   697   |  681 |      0       |  2062 |
| Test  | 1018 |   1044  |  944 |      0       |  3006 |
| Total | 7297 |   4296  | 6952 |      0       | 18545 |

### WTWT
WTWT splits are found from [Cross-Domain Label-Adaptive Stance Detection](https://arxiv.org/pdf/2104.07467.pdf)

|       | Comment | Refute | Support | Unrelated | Total |
|-------|:-------:|:------:|:-------:|:---------:|:-----:|
| Train |   8825  |  2761  |   3863  |   7550    | 22999 |
| Dev   |   2145  |   722  |   829   |   2435    |  6131 |
| Test  |   6887  |   325  |   1215  |   6019    | 14446 |
| Total |  17857  |  3808  |   5907  |   16004   | 43576 |



## Non-English

### ANS
ANS splits are found from the arabic news stance repo
|       | Contradiction Of | Discussing | Paraphrase Of | Unrelated To |
|-------|:----------------:|:----------:|:-------------:|:------------:|
| Train |       1686       |     63     |      903      |      0       |
| Dev   |        471       |     16     |      268      |      0       |
| Test  |        242       |     7      |      130      |      0       |

### NLPCC
NLPCC splits are generated by me and topics are shared between train, dev and test

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |   451   |      0     |    449   |      218     |
| Dev   |   161   |      0     |    140   |      73      |
| Test  |   597   |      0     |    575   |      321     |

### ASAP
ASAP splits are found from [ASAP](https://github.com/Meituan-Dianping/asap)

|       | Positive | Neutral | Negative | Not Mentioned |
|-------|:--------:|:-------:|:--------:|:-------------:|
| Train |  133721  |  52225  |   27425  |     449929    |
| Dev   |   18176  |   7192  |    3733  |     59819     |
| Test  |   17523  |   7026  |    3813  |     60558     |
