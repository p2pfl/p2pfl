# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/pguijas/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| p2pfl/\_\_main\_\_.py                                        |        3 |        3 |      0% |     21-24 |
| p2pfl/cli.py                                                 |       63 |       63 |      0% |    21-193 |
| p2pfl/commands/add\_model\_command.py                        |       39 |       12 |     69% |28, 69, 103-116 |
| p2pfl/commands/command.py                                    |        8 |        2 |     75% |    30, 35 |
| p2pfl/commands/heartbeat\_command.py                         |       13 |        1 |     92% |        42 |
| p2pfl/commands/init\_model\_command.py                       |       42 |       13 |     69% |28, 69-70, 76-80, 84-88, 102-103, 109-114 |
| p2pfl/commands/metrics\_command.py                           |       16 |        5 |     69% |     41-46 |
| p2pfl/commands/model\_initialized\_command.py                |       11 |        0 |    100% |           |
| p2pfl/commands/models\_agregated\_command.py                 |       13 |        0 |    100% |           |
| p2pfl/commands/models\_ready\_command.py                     |       15 |        1 |     93% |        49 |
| p2pfl/commands/start\_learning\_command.py                   |       13 |        1 |     92% |        49 |
| p2pfl/commands/stop\_learning\_command.py                    |       20 |        9 |     55% | 26, 46-57 |
| p2pfl/commands/vote\_train\_set\_command.py                  |       24 |        2 |     92% |     61-66 |
| p2pfl/communication/client.py                                |       15 |        4 |     73% |50, 73, 83, 92 |
| p2pfl/communication/communication\_protocol.py               |       46 |       14 |     70% |33, 38, 43, 48, 53, 60, 65, 70, 75, 80, 85, 90, 95, 108 |
| p2pfl/communication/gossiper.py                              |       96 |       14 |     85% |96, 123-125, 159, 165-166, 181-193 |
| p2pfl/communication/grpc/address.py                          |       53 |       23 |     57% |39-40, 46-49, 62-65, 74-76, 81-83, 87, 92-102 |
| p2pfl/communication/grpc/grpc\_client.py                     |       60 |        7 |     88% |71, 105, 147-149, 158-164 |
| p2pfl/communication/grpc/grpc\_communication\_protocol.py    |       65 |        5 |     92% |44-46, 115, 120, 132 |
| p2pfl/communication/grpc/grpc\_neighbors.py                  |       47 |        4 |     91% | 69, 78-80 |
| p2pfl/communication/grpc/grpc\_server.py                     |       71 |       15 |     79% |71-72, 81, 92, 121-124, 146-153, 168 |
| p2pfl/communication/grpc/proto/\_\_init\_\_.py               |        0 |        0 |    100% |           |
| p2pfl/communication/grpc/proto/node\_pb2.py                  |       22 |       11 |     50% |     24-34 |
| p2pfl/communication/grpc/proto/node\_pb2\_grpc.py            |       57 |       19 |     67% |18-19, 22, 70-72, 76-78, 82-84, 88-90, 137, 164, 191, 218 |
| p2pfl/communication/heartbeater.py                           |       45 |        1 |     98% |        75 |
| p2pfl/communication/neighbors.py                             |       50 |        8 |     84% |38, 42, 46, 55-56, 63-65 |
| p2pfl/learning/aggregators/aggregator.py                     |      106 |       20 |     81% |60, 79, 131-133, 137-143, 190, 225-232, 238-239 |
| p2pfl/learning/aggregators/fedavg.py                         |       18 |        1 |     94% |        46 |
| p2pfl/learning/exceptions.py                                 |        4 |        0 |    100% |           |
| p2pfl/learning/learner.py                                    |       28 |       13 |     54% |29, 44, 55, 72, 87, 102, 117, 128, 139, 143, 147, 151, 162 |
| p2pfl/learning/pytorch/lightning\_learner.py                 |       75 |       23 |     69% |77, 81, 106-107, 126, 132-146, 150-152, 158-170, 173-178 |
| p2pfl/learning/pytorch/lightning\_logger.py                  |       21 |        7 |     67% |37, 42, 46, 50-51, 55, 59 |
| p2pfl/learning/pytorch/mnist\_examples/mnistfederated\_dm.py |       48 |        9 |     81% |74-76, 85-87, 89, 115, 144 |
| p2pfl/learning/pytorch/mnist\_examples/models/cnn.py         |       59 |       33 |     44% |45-46, 50, 77-87, 91, 95-98, 102-109, 113-120 |
| p2pfl/learning/pytorch/mnist\_examples/models/mlp.py         |       53 |       33 |     38% |43-44, 49, 59-69, 73, 77-80, 84-91, 95-102 |
| p2pfl/management/logger.py                                   |      191 |       58 |     70% |46-54, 62-64, 69-71, 133-139, 159-160, 197-209, 239, 251, 263, 328, 350-353, 380-406, 422-424, 441, 458, 481-485, 492, 507, 514, 518, 548 |
| p2pfl/management/metric\_storage.py                          |       52 |       28 |     46% |64-85, 96, 111, 127, 144, 173-192, 203, 218, 234 |
| p2pfl/management/node\_monitor.py                            |       38 |       26 |     32% |36-45, 49, 53-58, 62-75, 80 |
| p2pfl/management/p2pfl\_web\_services.py                     |       75 |       60 |     20% |42-44, 60-65, 68-70, 83-95, 106, 121-144, 161-186, 202-226, 241-261, 265 |
| p2pfl/node.py                                                |      110 |       28 |     75% |203, 227-228, 269-272, 287-290, 318, 335, 339-347, 366-370, 373-384 |
| p2pfl/node\_state.py                                         |       36 |        1 |     97% |        70 |
| p2pfl/settings.py                                            |       35 |       16 |     54% |33, 37, 41, 49, 53, 61, 65, 69, 73, 77, 81, 85, 93, 97, 101, 105 |
| p2pfl/stages/base\_node/gossip\_model\_stage.py              |       57 |        6 |     89% |50, 71, 82, 98, 111, 113 |
| p2pfl/stages/base\_node/round\_finished\_stage.py            |       43 |        8 |     81% |50, 68, 77-78, 84, 89-91 |
| p2pfl/stages/base\_node/start\_learning\_stage.py            |       57 |        5 |     91% |65, 91-92, 115, 117 |
| p2pfl/stages/base\_node/train\_stage.py                      |       73 |        9 |     88% |51, 69, 92, 99, 104-106, 158, 160 |
| p2pfl/stages/base\_node/vote\_train\_set\_stage.py           |       84 |        4 |     95% |50, 107, 133-136 |
| p2pfl/stages/base\_node/wait\_agg\_models\_stage.py          |       18 |        2 |     89% |    43, 45 |
| p2pfl/stages/stage.py                                        |        8 |        2 |     75% |    29, 34 |
| p2pfl/stages/stage\_factory.py                               |       24 |        1 |     96% |        59 |
| p2pfl/stages/workflows.py                                    |       21 |        2 |     90% |    40, 47 |
| p2pfl/utils.py                                               |       54 |        5 |     91% |28, 59, 66, 77, 93 |
|                                                    **TOTAL** | **2295** |  **637** | **72%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/pguijas/p2pfl/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/pguijas/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pguijas/p2pfl/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/pguijas/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fpguijas%2Fp2pfl%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/pguijas/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.