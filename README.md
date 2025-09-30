# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/p2pfl/p2pfl/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                          |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| p2pfl/\_\_init\_\_.py                                                         |        0 |        0 |    100% |           |
| p2pfl/\_\_main\_\_.py                                                         |        3 |        3 |      0% |     21-24 |
| p2pfl/communication/\_\_init\_\_.py                                           |        0 |        0 |    100% |           |
| p2pfl/communication/commands/\_\_init\_\_.py                                  |        0 |        0 |    100% |           |
| p2pfl/communication/commands/command.py                                       |        8 |        2 |     75% |    30, 43 |
| p2pfl/communication/commands/message/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| p2pfl/communication/commands/message/heartbeat\_command.py                    |       12 |        1 |     92% |        49 |
| p2pfl/communication/commands/message/metrics\_command.py                      |       15 |        0 |    100% |           |
| p2pfl/communication/commands/message/model\_initialized\_command.py           |       11 |        0 |    100% |           |
| p2pfl/communication/commands/message/models\_agregated\_command.py            |       16 |        1 |     94% |        57 |
| p2pfl/communication/commands/message/models\_ready\_command.py                |       15 |        1 |     93% |        57 |
| p2pfl/communication/commands/message/pre\_send\_model\_command.py             |       48 |        6 |     88% |45-47, 52, 66, 68 |
| p2pfl/communication/commands/message/start\_learning\_command.py              |       13 |        1 |     92% |        63 |
| p2pfl/communication/commands/message/stop\_learning\_command.py               |       22 |        7 |     68% |     54-64 |
| p2pfl/communication/commands/message/vote\_train\_set\_command.py             |       24 |        2 |     92% |     69-74 |
| p2pfl/communication/commands/weights/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| p2pfl/communication/commands/weights/full\_model\_command.py                  |       40 |       15 |     62% |55, 61-65, 67-68, 77-89 |
| p2pfl/communication/commands/weights/init\_model\_command.py                  |       44 |       18 |     59% |55-56, 62-66, 70-74, 83-104 |
| p2pfl/communication/commands/weights/partial\_model\_command.py               |       46 |       16 |     65% |68, 74-78, 82-83, 100-116 |
| p2pfl/communication/protocols/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| p2pfl/communication/protocols/communication\_protocol.py                      |       47 |       12 |     74% |47, 52, 64, 77, 92, 112, 124, 136, 148, 159, 174, 198 |
| p2pfl/communication/protocols/exceptions.py                                   |        6 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/\_\_init\_\_.py                       |        0 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/client.py                             |       29 |        4 |     86% |56, 61, 66, 123 |
| p2pfl/communication/protocols/protobuff/gossiper.py                           |      111 |       14 |     87% |118, 126, 151-153, 201-202, 217-229 |
| p2pfl/communication/protocols/protobuff/grpc/\_\_init\_\_.py                  |       14 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/grpc/address.py                       |       52 |       23 |     56% |44-45, 51-54, 67-70, 79-81, 92-94, 98, 103-113 |
| p2pfl/communication/protocols/protobuff/grpc/client.py                        |       88 |       30 |     66% |67-68, 86, 91, 100-102, 115-116, 125-126, 153-162, 181-184, 188, 191, 198-201, 203, 207 |
| p2pfl/communication/protocols/protobuff/grpc/server.py                        |       40 |        4 |     90% |100-102, 113 |
| p2pfl/communication/protocols/protobuff/heartbeater.py                        |       49 |        1 |     98% |        79 |
| p2pfl/communication/protocols/protobuff/memory/\_\_init\_\_.py                |       14 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/memory/client.py                      |       63 |       12 |     81% |62-63, 76-77, 82-86, 92-93, 100-101, 157 |
| p2pfl/communication/protocols/protobuff/memory/server.py                      |       40 |        2 |     95% |  106, 119 |
| p2pfl/communication/protocols/protobuff/memory/singleton\_dict.py             |        3 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/neighbors.py                          |       61 |        6 |     90% |77-78, 84-86, 163 |
| p2pfl/communication/protocols/protobuff/proto/\_\_init\_\_.py                 |        0 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/proto/generate\_proto.py              |       23 |       23 |      0% |     23-70 |
| p2pfl/communication/protocols/protobuff/proto/node\_pb2.py                    |       28 |       15 |     46% |     34-48 |
| p2pfl/communication/protocols/protobuff/proto/node\_pb2\_grpc.py              |       47 |       15 |     68% |16-17, 20, 60-62, 66-68, 72-74, 116, 143, 170 |
| p2pfl/communication/protocols/protobuff/protobuff\_communication\_protocol.py |      109 |        4 |     96% |91, 97, 258, 297 |
| p2pfl/communication/protocols/protobuff/server.py                             |       74 |       13 |     82% |80, 85, 90, 101, 119, 178-184, 217 |
| p2pfl/examples/\_\_init\_\_.py                                                |        0 |        0 |    100% |           |
| p2pfl/examples/mnist/model/mlp\_flax.py                                       |       14 |        6 |     57% | 52-56, 62 |
| p2pfl/examples/mnist/model/mlp\_pytorch.py                                    |       61 |        7 |     89% |53, 74-79, 107 |
| p2pfl/examples/mnist/model/mlp\_tensorflow.py                                 |       29 |        0 |    100% |           |
| p2pfl/exceptions.py                                                           |        6 |        0 |    100% |           |
| p2pfl/experiment.py                                                           |       43 |       24 |     44% |62, 77-80, 93-108, 112-128 |
| p2pfl/learning/\_\_init\_\_.py                                                |        0 |        0 |    100% |           |
| p2pfl/learning/aggregators/\_\_init\_\_.py                                    |        2 |        0 |    100% |           |
| p2pfl/learning/aggregators/aggregator.py                                      |       97 |       20 |     79% |56, 77, 107, 145-147, 181-196, 218, 221, 274-278, 291 |
| p2pfl/learning/aggregators/fedavg.py                                          |       21 |        0 |    100% |           |
| p2pfl/learning/aggregators/fedmedian.py                                       |       17 |       17 |      0% |     21-68 |
| p2pfl/learning/aggregators/fedopt/\_\_init\_\_.py                             |        5 |        0 |    100% |           |
| p2pfl/learning/aggregators/fedopt/base.py                                     |       33 |       20 |     39% |53-62, 77-94, 114-119, 137, 148 |
| p2pfl/learning/aggregators/fedopt/fedadagrad.py                               |       13 |        8 |     38% |47-50, 64-79 |
| p2pfl/learning/aggregators/fedopt/fedadam.py                                  |       18 |       13 |     28% |55-61, 75-101 |
| p2pfl/learning/aggregators/fedopt/fedyogi.py                                  |       14 |        9 |     36% |55-61, 75-91 |
| p2pfl/learning/aggregators/fedprox.py                                         |       14 |       14 |      0% |     21-79 |
| p2pfl/learning/aggregators/krum.py                                            |       38 |       38 |      0% |    21-103 |
| p2pfl/learning/aggregators/scaffold.py                                        |       54 |        4 |     93% |87, 95, 108, 126 |
| p2pfl/learning/compression/\_\_init\_\_.py                                    |        7 |        0 |    100% |           |
| p2pfl/learning/compression/base\_compression\_strategy.py                     |       24 |        6 |     75% |33, 38, 47, 52, 61, 66 |
| p2pfl/learning/compression/dp\_strategy.py                                    |       43 |        6 |     86% |87-88, 125-127, 133 |
| p2pfl/learning/compression/lra\_strategy.py                                   |       25 |        0 |    100% |           |
| p2pfl/learning/compression/lzma\_strategy.py                                  |        7 |        0 |    100% |           |
| p2pfl/learning/compression/manager.py                                         |       51 |        2 |     96% |  111, 116 |
| p2pfl/learning/compression/quantization\_strategy.py                          |      223 |       72 |     68% |89-90, 95, 98, 166, 169, 189, 192, 197, 203, 237-244, 248, 314, 317, 322-323, 328-337, 344, 364-366, 384-415, 445, 448, 451, 455, 490, 493, 496, 499, 502, 506, 510, 514, 517, 522, 540, 543, 546, 554 |
| p2pfl/learning/compression/topk\_strategy.py                                  |       28 |        1 |     96% |        82 |
| p2pfl/learning/compression/zlib\_strategy.py                                  |        7 |        0 |    100% |           |
| p2pfl/learning/dataset/\_\_init\_\_.py                                        |        0 |        0 |    100% |           |
| p2pfl/learning/dataset/p2pfl\_dataset.py                                      |       88 |       25 |     72% |53, 140, 155, 159-163, 185-196, 210, 215, 299-300, 315-316, 331-332, 346-347, 377-378 |
| p2pfl/learning/dataset/partition\_strategies.py                               |      103 |       12 |     88% |58, 139, 190-195, 197, 270, 421, 423, 425 |
| p2pfl/learning/frameworks/\_\_init\_\_.py                                     |        5 |        0 |    100% |           |
| p2pfl/learning/frameworks/callback.py                                         |       13 |        1 |     92% |        42 |
| p2pfl/learning/frameworks/callback\_factory.py                                |       40 |       15 |     62% |51, 69-80, 91-92, 98-99, 105-106 |
| p2pfl/learning/frameworks/exceptions.py                                       |        4 |        0 |    100% |           |
| p2pfl/learning/frameworks/flax/\_\_init\_\_.py                                |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/flax/flax\_dataset.py                               |       16 |        7 |     56% |     57-65 |
| p2pfl/learning/frameworks/flax/flax\_learner.py                               |       86 |       86 |      0% |    21-181 |
| p2pfl/learning/frameworks/flax/flax\_model.py                                 |       54 |       34 |     37% |53-58, 62, 66-69, 79, 92-101, 111-123, 133-140, 153-154, 164 |
| p2pfl/learning/frameworks/learner.py                                          |       73 |        9 |     88% |51, 56, 59, 88, 112, 158, 163, 174, 185 |
| p2pfl/learning/frameworks/learner\_factory.py                                 |       19 |        5 |     74% |     48-54 |
| p2pfl/learning/frameworks/p2pfl\_model.py                                     |       58 |        9 |     84% |64, 99-100, 110, 123, 147, 164, 170, 194 |
| p2pfl/learning/frameworks/pytorch/\_\_init\_\_.py                             |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/pytorch/callbacks/\_\_init\_\_.py                   |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/pytorch/callbacks/fedprox\_callback.py              |       32 |       21 |     34% |43-45, 50, 62-68, 82-103 |
| p2pfl/learning/frameworks/pytorch/callbacks/scaffold\_callback.py             |       62 |       45 |     27% |41-47, 52, 63-74, 87-88, 103-110, 121-140, 143, 147-150 |
| p2pfl/learning/frameworks/pytorch/lightning\_dataset.py                       |       26 |        2 |     92% |   99, 105 |
| p2pfl/learning/frameworks/pytorch/lightning\_learner.py                       |       73 |       14 |     81% |75, 79, 85, 109-115, 119-121, 143-149 |
| p2pfl/learning/frameworks/pytorch/lightning\_logger.py                        |       22 |        2 |     91% |    45, 54 |
| p2pfl/learning/frameworks/pytorch/lightning\_model.py                         |       27 |        2 |     93% |     98-99 |
| p2pfl/learning/frameworks/simulation/\_\_init\_\_.py                          |        7 |        0 |    100% |           |
| p2pfl/learning/frameworks/simulation/actor\_pool.py                           |      139 |       34 |     76% |45-46, 50-56, 60-66, 153-163, 175, 187-191, 212, 266-267, 284-288, 338, 343, 366-367 |
| p2pfl/learning/frameworks/simulation/virtual\_learner.py                      |       51 |       10 |     80% |105, 109, 121-123, 128, 145-147, 151 |
| p2pfl/learning/frameworks/tensorflow/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/tensorflow/callbacks/\_\_init\_\_.py                |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/tensorflow/callbacks/keras\_logger.py               |       20 |        0 |    100% |           |
| p2pfl/learning/frameworks/tensorflow/callbacks/scaffold\_callback.py          |       69 |       51 |     26% |44-47, 52-59, 63, 78-84, 89, 93-116, 132, 142-159, 163 |
| p2pfl/learning/frameworks/tensorflow/keras\_dataset.py                        |       13 |        1 |     92% |        51 |
| p2pfl/learning/frameworks/tensorflow/keras\_learner.py                        |       62 |       11 |     82% |77, 81, 103-105, 111, 120, 126-129 |
| p2pfl/learning/frameworks/tensorflow/keras\_model.py                          |       25 |        1 |     96% |        69 |
| p2pfl/management/\_\_init\_\_.py                                              |        0 |        0 |    100% |           |
| p2pfl/management/cli.py                                                       |       90 |       90 |      0% |    21-301 |
| p2pfl/management/launch\_from\_yaml.py                                        |      159 |      159 |      0% |    20-300 |
| p2pfl/management/logger/\_\_init\_\_.py                                       |       15 |        1 |     93% |        43 |
| p2pfl/management/logger/decorators/async\_logger.py                           |       22 |       12 |     45% |34-46, 51, 55-58 |
| p2pfl/management/logger/decorators/file\_logger.py                            |       38 |       29 |     24% |35-38, 42-89 |
| p2pfl/management/logger/decorators/logger\_decorator.py                       |       53 |       22 |     58% |47, 59, 63, 67, 77, 87, 100, 112, 126, 140, 154, 164, 174, 185, 196, 206, 216, 226, 253, 286, 296, 305 |
| p2pfl/management/logger/decorators/ray\_logger.py                             |       74 |       17 |     77% |48, 79-81, 93, 97, 101, 147, 202, 230, 285, 296, 306, 316, 340, 389, 398 |
| p2pfl/management/logger/decorators/singleton\_logger.py                       |        4 |        0 |    100% |           |
| p2pfl/management/logger/decorators/wandb\_logger.py                           |       96 |       82 |     15% |28-30, 42-56, 84-116, 128-174, 179-192, 196-204 |
| p2pfl/management/logger/decorators/web\_logger.py                             |       93 |       70 |     25% |48-56, 70-72, 83-85, 98-105, 110-126, 144-165, 177-182, 196-211, 239-270, 280-282, 292-294, 303 |
| p2pfl/management/logger/logger.py                                             |      163 |      115 |     29% |74-85, 105-130, 143, 148-153, 162, 176-179, 189, 202, 213, 224, 235, 246, 257, 270-281, 300-323, 337, 351, 366-369, 380-384, 399, 410-414, 424-425, 435, 445, 477-511, 546-552, 562, 572-580 |
| p2pfl/management/message\_storage.py                                          |       57 |       44 |     23% |55-56, 84-121, 146-185, 203, 221 |
| p2pfl/management/metric\_storage.py                                           |       55 |       36 |     35% |51-52, 76-99, 109, 122, 136, 151, 176-177, 192-213, 223, 236, 250 |
| p2pfl/management/node\_monitor.py                                             |       34 |       22 |     35% |42-50, 54, 58, 62, 66-76, 80-85 |
| p2pfl/management/p2pfl\_web\_services.py                                      |       80 |       65 |     19% |52-54, 69-74, 78-80, 91-102, 112, 126-150, 166-192, 207-232, 246-267, 296, 300 |
| p2pfl/node.py                                                                 |      138 |       36 |     74% |187-190, 214, 237-238, 258-259, 276-278, 291-293, 307-309, 333, 365, 384-385, 389-395, 430-432, 435-446 |
| p2pfl/node\_state.py                                                          |       46 |        3 |     93% |93, 146, 158 |
| p2pfl/settings.py                                                             |       94 |        9 |     90% |   153-164 |
| p2pfl/stages/\_\_init\_\_.py                                                  |        0 |        0 |    100% |           |
| p2pfl/stages/base\_node/\_\_init\_\_.py                                       |        0 |        0 |    100% |           |
| p2pfl/stages/base\_node/gossip\_model\_stage.py                               |       37 |        3 |     92% |50, 64, 77 |
| p2pfl/stages/base\_node/round\_finished\_stage.py                             |       36 |        2 |     94% |    49, 61 |
| p2pfl/stages/base\_node/start\_learning\_stage.py                             |       50 |        3 |     94% |64, 94, 120 |
| p2pfl/stages/base\_node/train\_stage.py                                       |       78 |        4 |     95% |53, 100-101, 156 |
| p2pfl/stages/base\_node/vote\_train\_set\_stage.py                            |       86 |        6 |     93% |51, 74-75, 142-143, 184 |
| p2pfl/stages/base\_node/wait\_agg\_models\_stage.py                           |       24 |        2 |     92% |    43, 55 |
| p2pfl/stages/stage.py                                                         |       18 |        6 |     67% |30, 35, 60-63 |
| p2pfl/stages/stage\_factory.py                                                |       23 |        1 |     96% |        57 |
| p2pfl/stages/workflows.py                                                     |       25 |        1 |     96% |        50 |
| p2pfl/utils/check\_ray.py                                                     |       13 |        2 |     85% |    30, 48 |
| p2pfl/utils/node\_component.py                                                |       32 |        0 |    100% |           |
| p2pfl/utils/seed.py                                                           |       33 |       12 |     64% |57, 59-60, 63-71, 79-80 |
| p2pfl/utils/singleton.py                                                      |        7 |        0 |    100% |           |
| p2pfl/utils/topologies.py                                                     |       71 |        4 |     94% |97, 109, 131-132 |
| p2pfl/utils/utils.py                                                          |      113 |       16 |     86% |96, 100, 126, 137-139, 157-160, 178, 188, 200-201, 221, 230, 253 |
|                                                                     **TOTAL** | **5253** | **1789** | **66%** |           |


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