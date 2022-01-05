# ml-stance-detection

English datasets are from [mdl-stance-robustness](https://github.com/UKPLab/mdl-stance-robustness) \
NLPCC dataset downloaded from http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html \
I am using the Task A Annotated Training Data from stance detection in chinese microblogs task

## Preliminary Results
Train on nlpcc (zh) and test on nlpcc (zh), acc = ~63% \
Train on arc (en) and test on arc (en), acc = ~75%

## Datasets
### NLPCC
NLPCC splits are generated by me and topics are shared between train, dev and test

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |   451   |      0     |    449   |      218     |
| Dev   |   161   |      0     |    140   |      73      |
| Test  |   597   |      0     |    575   |      321     |

### ARC
ARC splits are found from the mdl repo and topics are shared


|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |   1220  |     786    |   1094   |     9282     |
| Dev   |   182   |     118    |    163   |     1388     |
| Test  |   372   |     179    |    334   |     2674     |
