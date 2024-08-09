# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/pguijas/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                          |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| p2pfl/\_\_init\_\_.py                                         |        0 |        0 |    100% |           |
| p2pfl/\_\_main\_\_.py                                         |        3 |        3 |      0% |     21-24 |
| p2pfl/cli.py                                                  |       64 |       64 |      0% |    21-207 |
| p2pfl/commands/\_\_init\_\_.py                                |        0 |        0 |    100% |           |
| p2pfl/commands/add\_model\_command.py                         |       38 |       11 |     71% |61, 95-108 |
| p2pfl/commands/command.py                                     |        8 |        2 |     75% |    30, 43 |
| p2pfl/commands/heartbeat\_command.py                          |       13 |        1 |     92% |        51 |
| p2pfl/commands/init\_model\_command.py                        |       41 |       12 |     71% |72-73, 79-83, 87-91, 105-106, 112-117 |
| p2pfl/commands/metrics\_command.py                            |       16 |        5 |     69% |     50-55 |
| p2pfl/commands/model\_initialized\_command.py                 |       11 |        0 |    100% |           |
| p2pfl/commands/models\_agregated\_command.py                  |       13 |        0 |    100% |           |
| p2pfl/commands/models\_ready\_command.py                      |       15 |        1 |     93% |        57 |
| p2pfl/commands/start\_learning\_command.py                    |       13 |        1 |     92% |        59 |
| p2pfl/commands/stop\_learning\_command.py                     |       19 |        8 |     58% |     50-61 |
| p2pfl/commands/vote\_train\_set\_command.py                   |       24 |        2 |     92% |     69-74 |
| p2pfl/communication/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| p2pfl/communication/client.py                                 |       15 |        4 |     73% |48, 70, 80, 89 |
| p2pfl/communication/communication\_protocol.py                |       46 |       14 |     70% |40, 45, 50, 61, 74, 91, 103, 115, 127, 139, 150, 161, 166, 190 |
| p2pfl/communication/exceptions.py                             |        2 |        0 |    100% |           |
| p2pfl/communication/gossiper.py                               |       96 |       14 |     85% |118, 145-147, 192, 198-199, 214-226 |
| p2pfl/communication/grpc/\_\_init\_\_.py                      |        0 |        0 |    100% |           |
| p2pfl/communication/grpc/address.py                           |       53 |       23 |     57% |45-46, 52-55, 68-71, 80-82, 93-95, 99, 104-114 |
| p2pfl/communication/grpc/grpc\_client.py                      |       59 |        7 |     88% |71, 104, 154-156, 165-171 |
| p2pfl/communication/grpc/grpc\_communication\_protocol.py     |       62 |        3 |     95% |155, 167, 198 |
| p2pfl/communication/grpc/grpc\_neighbors.py                   |       46 |        4 |     91% | 83, 92-94 |
| p2pfl/communication/grpc/grpc\_server.py                      |       71 |       15 |     79% |86-87, 96, 114, 157-160, 189-196, 217 |
| p2pfl/communication/grpc/proto/\_\_init\_\_.py                |        0 |        0 |    100% |           |
| p2pfl/communication/grpc/proto/generate\_proto.py             |       23 |       23 |      0% |     23-71 |
| p2pfl/communication/grpc/proto/node\_pb2.py                   |       22 |       11 |     50% |     24-34 |
| p2pfl/communication/grpc/proto/node\_pb2\_grpc.py             |       57 |       19 |     67% |18-19, 22, 70-72, 76-78, 82-84, 88-90, 137, 164, 191, 218 |
| p2pfl/communication/heartbeater.py                            |       44 |        1 |     98% |        73 |
| p2pfl/communication/memory/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| p2pfl/communication/memory/memory\_client.py                  |       45 |       45 |      0% |    20-172 |
| p2pfl/communication/memory/memory\_communication\_protocol.py |       59 |       59 |      0% |    20-230 |
| p2pfl/communication/memory/memory\_neighbors.py               |       36 |       36 |      0% |    20-109 |
| p2pfl/communication/memory/memory\_server.py                  |       64 |       64 |      0% |    20-204 |
| p2pfl/communication/memory/server\_singleton.py               |       16 |       16 |      0% |     19-43 |
| p2pfl/communication/neighbors.py                              |       50 |        8 |     84% |50, 60, 71, 88-89, 96-98 |
| p2pfl/examples/\_\_init\_\_.py                                |        0 |        0 |    100% |           |
| p2pfl/examples/mnist.py                                       |       79 |       79 |      0% |    21-178 |
| p2pfl/examples/node1.py                                       |       17 |       17 |      0% |     25-63 |
| p2pfl/examples/node2.py                                       |       23 |       23 |      0% |     25-77 |
| p2pfl/exceptions.py                                           |        6 |        0 |    100% |           |
| p2pfl/learning/\_\_init\_\_.py                                |        0 |        0 |    100% |           |
| p2pfl/learning/aggregators/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| p2pfl/learning/aggregators/aggregator.py                      |      106 |       20 |     81% |65, 82, 134-136, 140-146, 193, 225-232, 238-239 |
| p2pfl/learning/aggregators/fedavg.py                          |       16 |        1 |     94% |        41 |
| p2pfl/learning/exceptions.py                                  |        4 |        0 |    100% |           |
| p2pfl/learning/learner.py                                     |       26 |       12 |     54% |38, 51, 61, 76, 89, 102, 112, 122, 126, 130, 140, 150 |
| p2pfl/learning/pytorch/\_\_init\_\_.py                        |        0 |        0 |    100% |           |
| p2pfl/learning/pytorch/lightning\_learner.py                  |       75 |       23 |     69% |83, 93, 137-138, 178, 184-198, 202-204, 216-228, 231-236 |
| p2pfl/learning/pytorch/lightning\_logger.py                   |       21 |        7 |     67% |43, 48, 52, 56-57, 61, 65 |
| p2pfl/learning/pytorch/mnist\_examples/\_\_init\_\_.py        |        0 |        0 |    100% |           |
| p2pfl/learning/pytorch/mnist\_examples/mnistfederated\_dm.py  |       48 |        9 |     81% |88-90, 99-101, 103, 129, 158 |
| p2pfl/learning/pytorch/mnist\_examples/models/\_\_init\_\_.py |        0 |        0 |    100% |           |
| p2pfl/learning/pytorch/mnist\_examples/models/cnn.py          |       59 |       33 |     44% |45-46, 50, 77-87, 91, 95-98, 102-109, 113-120 |
| p2pfl/learning/pytorch/mnist\_examples/models/mlp.py          |       53 |       33 |     38% |43-44, 49, 59-69, 73, 77-80, 84-91, 95-102 |
| p2pfl/management/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| p2pfl/management/logger.py                                    |      202 |       66 |     67% |57-65, 79-81, 92-94, 129, 179-185, 205-206, 243-255, 283, 368, 389-392, 424-462, 477-479, 494, 509, 531-535, 542, 556, 563, 567, 593 |
| p2pfl/management/metric\_storage.py                           |       52 |       28 |     46% |77-98, 108, 121, 135, 150, 191-210, 220, 233, 247 |
| p2pfl/management/node\_monitor.py                             |       38 |       26 |     32% |43-52, 56, 60-65, 69-82, 86 |
| p2pfl/management/p2pfl\_web\_services.py                      |       75 |       60 |     20% |53-55, 70-75, 78-80, 92-104, 114, 128-151, 167-192, 207-231, 245-265, 269 |
| p2pfl/node.py                                                 |      114 |       28 |     75% |207, 229-230, 268-271, 284-287, 317, 334, 338-346, 365-369, 372-383 |
| p2pfl/node\_state.py                                          |       36 |        1 |     97% |       106 |
| p2pfl/settings.py                                             |       35 |       16 |     54% |33, 37, 41, 49, 53, 61, 65, 69, 73, 77, 81, 85, 93, 97, 101, 105 |
| p2pfl/stages/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| p2pfl/stages/base\_node/\_\_init\_\_.py                       |        0 |        0 |    100% |           |
| p2pfl/stages/base\_node/gossip\_model\_stage.py               |       57 |        6 |     89% |50, 71, 82, 98, 111, 113 |
| p2pfl/stages/base\_node/round\_finished\_stage.py             |       43 |        8 |     81% |50, 68, 77-78, 84, 89-91 |
| p2pfl/stages/base\_node/start\_learning\_stage.py             |       57 |        5 |     91% |65, 91-92, 115, 117 |
| p2pfl/stages/base\_node/train\_stage.py                       |       73 |        9 |     88% |51, 69, 92, 99, 104-106, 158, 160 |
| p2pfl/stages/base\_node/vote\_train\_set\_stage.py            |       84 |        4 |     95% |50, 107, 133-136 |
| p2pfl/stages/base\_node/wait\_agg\_models\_stage.py           |       18 |        2 |     89% |    43, 45 |
| p2pfl/stages/stage.py                                         |        8 |        2 |     75% |    29, 34 |
| p2pfl/stages/stage\_factory.py                                |       24 |        1 |     96% |        59 |
| p2pfl/stages/workflows.py                                     |       21 |        2 |     90% |    40, 47 |
| p2pfl/utils.py                                                |       54 |        5 |     91% |28, 71, 78, 102, 127 |
|                                                     **TOTAL** | **2668** | **1002** | **62%** |           |


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