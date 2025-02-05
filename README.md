# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/p2pfl/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                    |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| p2pfl/\_\_init\_\_.py                                                   |        0 |        0 |    100% |           |
| p2pfl/\_\_main\_\_.py                                                   |        3 |        3 |      0% |     21-24 |
| p2pfl/cli.py                                                            |       79 |       79 |      0% |    21-242 |
| p2pfl/communication/\_\_init\_\_.py                                     |        0 |        0 |    100% |           |
| p2pfl/communication/commands/\_\_init\_\_.py                            |        0 |        0 |    100% |           |
| p2pfl/communication/commands/command.py                                 |        8 |        2 |     75% |    30, 43 |
| p2pfl/communication/commands/message/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| p2pfl/communication/commands/message/heartbeat\_command.py              |       13 |        1 |     92% |        51 |
| p2pfl/communication/commands/message/metrics\_command.py                |       15 |        0 |    100% |           |
| p2pfl/communication/commands/message/model\_initialized\_command.py     |       11 |        0 |    100% |           |
| p2pfl/communication/commands/message/models\_agregated\_command.py      |       13 |        1 |     92% |        53 |
| p2pfl/communication/commands/message/models\_ready\_command.py          |       15 |        1 |     93% |        57 |
| p2pfl/communication/commands/message/start\_learning\_command.py        |       13 |        1 |     92% |        59 |
| p2pfl/communication/commands/message/stop\_learning\_command.py         |       22 |        7 |     68% |     54-64 |
| p2pfl/communication/commands/message/vote\_train\_set\_command.py       |       24 |        2 |     92% |     69-74 |
| p2pfl/communication/commands/weights/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| p2pfl/communication/commands/weights/full\_model\_command.py            |       40 |       13 |     68% |55, 61-65, 77-89 |
| p2pfl/communication/commands/weights/init\_model\_command.py            |       41 |       16 |     61% |55-56, 62-66, 70-74, 84-97 |
| p2pfl/communication/commands/weights/partial\_model\_command.py         |       44 |       15 |     66% |67, 73-77, 81-82, 99-112 |
| p2pfl/communication/protocols/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| p2pfl/communication/protocols/client.py                                 |       15 |        4 |     73% |48, 70, 80, 89 |
| p2pfl/communication/protocols/communication\_protocol.py                |       46 |       14 |     70% |40, 45, 50, 61, 74, 91, 111, 123, 135, 147, 158, 169, 174, 198 |
| p2pfl/communication/protocols/exceptions.py                             |        6 |        0 |    100% |           |
| p2pfl/communication/protocols/gossiper.py                               |       95 |       12 |     87% |144-146, 194-195, 210-222 |
| p2pfl/communication/protocols/grpc/\_\_init\_\_.py                      |        0 |        0 |    100% |           |
| p2pfl/communication/protocols/grpc/address.py                           |       53 |       23 |     57% |45-46, 52-55, 68-71, 80-82, 93-95, 99, 104-114 |
| p2pfl/communication/protocols/grpc/grpc\_client.py                      |       61 |        4 |     93% |70, 108, 165, 173 |
| p2pfl/communication/protocols/grpc/grpc\_communication\_protocol.py     |       79 |        1 |     99% |       230 |
| p2pfl/communication/protocols/grpc/grpc\_neighbors.py                   |       56 |        5 |     91% |97, 101, 110-112 |
| p2pfl/communication/protocols/grpc/grpc\_server.py                      |       85 |       13 |     85% |106-108, 119, 147, 198-204, 237 |
| p2pfl/communication/protocols/grpc/proto/\_\_init\_\_.py                |        0 |        0 |    100% |           |
| p2pfl/communication/protocols/grpc/proto/generate\_proto.py             |       23 |       23 |      0% |     23-70 |
| p2pfl/communication/protocols/grpc/proto/node\_pb2.py                   |       24 |       13 |     46% |     24-36 |
| p2pfl/communication/protocols/grpc/proto/node\_pb2\_grpc.py             |       35 |       12 |     66% |40-42, 46-48, 52-54, 95, 112, 129 |
| p2pfl/communication/protocols/heartbeater.py                            |       44 |        1 |     98% |        75 |
| p2pfl/communication/protocols/memory/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| p2pfl/communication/protocols/memory/memory\_client.py                  |       48 |        6 |     88% |65, 96-98, 135, 141 |
| p2pfl/communication/protocols/memory/memory\_communication\_protocol.py |       79 |        9 |     89% |68, 178-180, 219, 235, 259-261 |
| p2pfl/communication/protocols/memory/memory\_neighbors.py               |       38 |        2 |     95% |     79-80 |
| p2pfl/communication/protocols/memory/memory\_server.py                  |       74 |       18 |     76% |85, 98, 125, 166-169, 188-204, 220 |
| p2pfl/communication/protocols/memory/server\_singleton.py               |       16 |        1 |     94% |        43 |
| p2pfl/communication/protocols/neighbors.py                              |       49 |        8 |     84% |50, 60, 71, 85-86, 93-95 |
| p2pfl/examples/\_\_init\_\_.py                                          |        0 |        0 |    100% |           |
| p2pfl/examples/mnist.py                                                 |      144 |      144 |      0% |    24-297 |
| p2pfl/examples/node1.py                                                 |       19 |       19 |      0% |     25-60 |
| p2pfl/examples/node2.py                                                 |       27 |       27 |      0% |     25-75 |
| p2pfl/exceptions.py                                                     |        6 |        0 |    100% |           |
| p2pfl/experiment.py                                                     |       15 |        5 |     67% |51, 67-70, 74 |
| p2pfl/learning/\_\_init\_\_.py                                          |        0 |        0 |    100% |           |
| p2pfl/learning/aggregators/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| p2pfl/learning/aggregators/aggregator.py                                |       88 |       14 |     84% |64, 126-128, 166-171, 197, 200, 253-257, 270 |
| p2pfl/learning/aggregators/fedavg.py                                    |       22 |        0 |    100% |           |
| p2pfl/learning/aggregators/fedmedian.py                                 |        7 |        7 |      0% |     21-47 |
| p2pfl/learning/aggregators/scaffold.py                                  |       54 |        4 |     93% |87, 95, 108, 126 |
| p2pfl/learning/dataset/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| p2pfl/learning/dataset/p2pfl\_dataset.py                                |       71 |       16 |     77% |52, 136, 150, 166, 180, 185, 263-264, 279-280, 295-296, 310-311, 341-342 |
| p2pfl/learning/dataset/partition\_strategies.py                         |      104 |       14 |     87% |57, 142, 193-200, 202, 273, 426, 428, 430 |
| p2pfl/learning/frameworks/\_\_init\_\_.py                               |        5 |        0 |    100% |           |
| p2pfl/learning/frameworks/callback.py                                   |       13 |        1 |     92% |        42 |
| p2pfl/learning/frameworks/callback\_factory.py                          |       36 |       13 |     64% |53, 71-82, 93-94, 100-101 |
| p2pfl/learning/frameworks/exceptions.py                                 |        4 |        0 |    100% |           |
| p2pfl/learning/frameworks/flax/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/flax/flax\_dataset.py                         |       17 |        1 |     94% |        56 |
| p2pfl/learning/frameworks/flax/flax\_learner.py                         |       85 |        8 |     91% |137-139, 162-165, 171 |
| p2pfl/learning/frameworks/flax/flax\_model.py                           |       64 |        3 |     95% |138-139, 163 |
| p2pfl/learning/frameworks/learner.py                                    |       56 |        7 |     88% |64, 100, 110, 140, 145, 156, 167 |
| p2pfl/learning/frameworks/learner\_factory.py                           |       20 |        8 |     60% |     46-56 |
| p2pfl/learning/frameworks/p2pfl\_model.py                               |       59 |        9 |     85% |63, 100-101, 111, 124, 148, 165, 171, 195 |
| p2pfl/learning/frameworks/pytorch/\_\_init\_\_.py                       |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/pytorch/callbacks/\_\_init\_\_.py             |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/pytorch/callbacks/scaffold\_callback.py       |       62 |       45 |     27% |41-47, 52, 63-74, 87-88, 103-110, 121-140, 143, 147-150 |
| p2pfl/learning/frameworks/pytorch/lightning\_dataset.py                 |       23 |        1 |     96% |       100 |
| p2pfl/learning/frameworks/pytorch/lightning\_learner.py                 |       66 |       13 |     80% |71, 75, 102-108, 112-114, 136-142 |
| p2pfl/learning/frameworks/pytorch/lightning\_logger.py                  |       21 |        2 |     90% |    43, 52 |
| p2pfl/learning/frameworks/pytorch/lightning\_model.py                   |       83 |       11 |     87% |97-98, 136-137, 141, 162-167, 195 |
| p2pfl/learning/frameworks/simulation/\_\_init\_\_.py                    |       12 |        0 |    100% |           |
| p2pfl/learning/frameworks/simulation/actor\_pool.py                     |      137 |       25 |     82% |45-46, 50-56, 60-66, 117, 138, 238-239, 256-260, 324, 329, 338, 355-356 |
| p2pfl/learning/frameworks/simulation/utils.py                           |       29 |        6 |     79% |44-45, 76, 84, 88-94 |
| p2pfl/learning/frameworks/simulation/virtual\_learner.py                |       46 |        8 |     83% |111-113, 118, 135-137, 141 |
| p2pfl/learning/frameworks/tensorflow/\_\_init\_\_.py                    |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/tensorflow/callbacks/\_\_init\_\_.py          |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/tensorflow/callbacks/keras\_logger.py         |       20 |        0 |    100% |           |
| p2pfl/learning/frameworks/tensorflow/callbacks/scaffold\_callback.py    |       69 |       51 |     26% |44-47, 52-59, 63, 78-84, 89, 93-116, 132, 142-159, 163 |
| p2pfl/learning/frameworks/tensorflow/keras\_dataset.py                  |       14 |        1 |     93% |        62 |
| p2pfl/learning/frameworks/tensorflow/keras\_learner.py                  |       54 |       11 |     80% |63, 67, 88-90, 96, 105, 111-114 |
| p2pfl/learning/frameworks/tensorflow/keras\_model.py                    |       48 |        4 |     92% |70, 113, 141, 168 |
| p2pfl/management/\_\_init\_\_.py                                        |        0 |        0 |    100% |           |
| p2pfl/management/logger/\_\_init\_\_.py                                 |       10 |        1 |     90% |        35 |
| p2pfl/management/logger/decorators/async\_logger.py                     |       22 |       12 |     45% |34-46, 51, 55-58 |
| p2pfl/management/logger/decorators/file\_logger.py                      |       17 |        0 |    100% |           |
| p2pfl/management/logger/decorators/logger\_decorator.py                 |       45 |        6 |     87% |56, 60, 133, 189, 220, 243 |
| p2pfl/management/logger/decorators/ray\_logger.py                       |       50 |        7 |     86% |63, 67, 140, 196, 227, 237, 250 |
| p2pfl/management/logger/decorators/singleton\_logger.py                 |        4 |        0 |    100% |           |
| p2pfl/management/logger/decorators/web\_logger.py                       |       61 |       39 |     36% |48-56, 70-72, 83-85, 110-113, 127-140, 153-155, 166-175, 185-196 |
| p2pfl/management/logger/logger.py                                       |      121 |       63 |     48% |71-79, 108-109, 131, 136-141, 155-158, 168, 181, 236, 249-260, 286-308, 322, 336, 352-356, 367-372, 387-388, 398, 409-410, 421, 431, 454 |
| p2pfl/management/metric\_storage.py                                     |       56 |       32 |     43% |77-100, 110, 123, 137, 152, 193-214, 224, 237, 251 |
| p2pfl/management/node\_monitor.py                                       |       38 |       26 |     32% |43-52, 56, 60-65, 69-82, 86 |
| p2pfl/management/p2pfl\_web\_services.py                                |       75 |       60 |     20% |53-55, 70-75, 79-81, 93-105, 115, 129-152, 168-193, 208-232, 246-266, 270 |
| p2pfl/node.py                                                           |      125 |       35 |     72% |181-184, 208, 231-232, 252-253, 270-272, 285-287, 301-303, 317, 327, 357, 372, 376-382, 398-400, 403-413 |
| p2pfl/node\_state.py                                                    |       42 |        3 |     93% |97, 120, 131 |
| p2pfl/settings.py                                                       |       52 |        0 |    100% |           |
| p2pfl/stages/\_\_init\_\_.py                                            |        0 |        0 |    100% |           |
| p2pfl/stages/base\_node/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| p2pfl/stages/base\_node/gossip\_model\_stage.py                         |       37 |        3 |     92% |50, 64, 77 |
| p2pfl/stages/base\_node/round\_finished\_stage.py                       |       38 |        2 |     95% |    51, 64 |
| p2pfl/stages/base\_node/start\_learning\_stage.py                       |       51 |        2 |     96% |   55, 102 |
| p2pfl/stages/base\_node/train\_stage.py                                 |       76 |        4 |     95% |53, 99-100, 160 |
| p2pfl/stages/base\_node/vote\_train\_set\_stage.py                      |       87 |        6 |     93% |50, 76-77, 141-142, 183 |
| p2pfl/stages/base\_node/wait\_agg\_models\_stage.py                     |       25 |        2 |     92% |    45, 57 |
| p2pfl/stages/stage.py                                                   |       19 |        6 |     68% |32, 37, 62-65 |
| p2pfl/stages/stage\_factory.py                                          |       24 |        1 |     96% |        59 |
| p2pfl/stages/workflows.py                                               |       26 |        1 |     96% |        52 |
| p2pfl/utils/check\_ray.py                                               |       11 |        2 |     82% |    29, 42 |
| p2pfl/utils/singleton.py                                                |        7 |        0 |    100% |           |
| p2pfl/utils/topologies.py                                               |       43 |        2 |     95% |     92-93 |
| p2pfl/utils/utils.py                                                    |       58 |        5 |     91% |84, 96-97, 116, 139 |
|                                                               **TOTAL** | **4087** | **1108** | **73%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/p2pfl/p2pfl/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/p2pfl/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/p2pfl/p2pfl/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/p2pfl/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fp2pfl%2Fp2pfl%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/p2pfl/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.