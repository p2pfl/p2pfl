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
| p2pfl/communication/commands/message/heartbeat\_command.py                    |       13 |        1 |     92% |        51 |
| p2pfl/communication/commands/message/metrics\_command.py                      |       15 |        0 |    100% |           |
| p2pfl/communication/commands/message/model\_initialized\_command.py           |       11 |        0 |    100% |           |
| p2pfl/communication/commands/message/models\_agregated\_command.py            |       13 |        1 |     92% |        53 |
| p2pfl/communication/commands/message/models\_ready\_command.py                |       15 |        1 |     93% |        57 |
| p2pfl/communication/commands/message/start\_learning\_command.py              |       13 |        1 |     92% |        63 |
| p2pfl/communication/commands/message/stop\_learning\_command.py               |       22 |        7 |     68% |     54-64 |
| p2pfl/communication/commands/message/vote\_train\_set\_command.py             |       24 |        2 |     92% |     69-74 |
| p2pfl/communication/commands/weights/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| p2pfl/communication/commands/weights/full\_model\_command.py                  |       40 |       11 |     72% | 55, 77-89 |
| p2pfl/communication/commands/weights/init\_model\_command.py                  |       41 |       16 |     61% |55-56, 62-66, 70-74, 84-97 |
| p2pfl/communication/commands/weights/partial\_model\_command.py               |       44 |       15 |     66% |67, 73-77, 81-82, 99-112 |
| p2pfl/communication/protocols/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| p2pfl/communication/protocols/communication\_protocol.py                      |       46 |       12 |     74% |46, 51, 63, 76, 93, 113, 125, 137, 149, 160, 175, 199 |
| p2pfl/communication/protocols/exceptions.py                                   |        6 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/\_\_init\_\_.py                       |        0 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/client.py                             |       23 |        4 |     83% |55, 60, 65, 99 |
| p2pfl/communication/protocols/protobuff/gossiper.py                           |      103 |       14 |     86% |112, 120, 145-147, 195-196, 211-223 |
| p2pfl/communication/protocols/protobuff/grpc/\_\_init\_\_.py                  |       15 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/grpc/address.py                       |       53 |       23 |     57% |45-46, 52-55, 68-71, 80-82, 93-95, 99, 104-114 |
| p2pfl/communication/protocols/protobuff/grpc/client.py                        |       87 |       35 |     60% |68-69, 85, 90, 99-101, 114-115, 124-125, 152-163, 168-182, 192-195, 197 |
| p2pfl/communication/protocols/protobuff/grpc/server.py                        |       41 |        4 |     90% |99-101, 112 |
| p2pfl/communication/protocols/protobuff/heartbeater.py                        |       49 |        1 |     98% |        79 |
| p2pfl/communication/protocols/protobuff/memory/\_\_init\_\_.py                |       15 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/memory/client.py                      |       63 |       15 |     76% |64-65, 78-79, 84-88, 94-95, 102-103, 136-139, 156 |
| p2pfl/communication/protocols/protobuff/memory/server.py                      |       41 |        2 |     95% |  107, 120 |
| p2pfl/communication/protocols/protobuff/memory/singleton\_dict.py             |        3 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/neighbors.py                          |       58 |        4 |     93% |77-78, 84-85 |
| p2pfl/communication/protocols/protobuff/proto/\_\_init\_\_.py                 |        0 |        0 |    100% |           |
| p2pfl/communication/protocols/protobuff/proto/generate\_proto.py              |       23 |       23 |      0% |     23-70 |
| p2pfl/communication/protocols/protobuff/proto/node\_pb2.py                    |       26 |       13 |     50% |     34-46 |
| p2pfl/communication/protocols/protobuff/proto/node\_pb2\_grpc.py              |       47 |       15 |     68% |16-17, 20, 60-62, 66-68, 72-74, 116, 143, 170 |
| p2pfl/communication/protocols/protobuff/protobuff\_communication\_protocol.py |      106 |        4 |     96% |90, 96, 246, 285 |
| p2pfl/communication/protocols/protobuff/server.py                             |       73 |       13 |     82% |82, 87, 92, 103, 121, 172-178, 211 |
| p2pfl/examples/\_\_init\_\_.py                                                |        0 |        0 |    100% |           |
| p2pfl/examples/mnist/model/mlp\_flax.py                                       |       15 |        6 |     60% | 54-58, 64 |
| p2pfl/examples/mnist/model/mlp\_pytorch.py                                    |       62 |        7 |     89% |55, 76-81, 109 |
| p2pfl/examples/mnist/model/mlp\_tensorflow.py                                 |       29 |        0 |    100% |           |
| p2pfl/exceptions.py                                                           |        6 |        0 |    100% |           |
| p2pfl/experiment.py                                                           |       15 |        5 |     67% |51, 66-69, 73 |
| p2pfl/learning/\_\_init\_\_.py                                                |        0 |        0 |    100% |           |
| p2pfl/learning/aggregators/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| p2pfl/learning/aggregators/aggregator.py                                      |       90 |       13 |     86% |67, 129-131, 170, 201, 204, 257-261, 274 |
| p2pfl/learning/aggregators/fedavg.py                                          |       22 |        0 |    100% |           |
| p2pfl/learning/aggregators/fedmedian.py                                       |        7 |        7 |      0% |     21-47 |
| p2pfl/learning/aggregators/scaffold.py                                        |       54 |        4 |     93% |86, 94, 107, 125 |
| p2pfl/learning/compression/\_\_init\_\_.py                                    |        6 |        0 |    100% |           |
| p2pfl/learning/compression/base\_compression\_strategy.py                     |       24 |        6 |     75% |33, 38, 47, 52, 61, 66 |
| p2pfl/learning/compression/lra\_strategy.py                                   |       25 |        0 |    100% |           |
| p2pfl/learning/compression/lzma\_strategy.py                                  |        7 |        0 |    100% |           |
| p2pfl/learning/compression/manager.py                                         |       51 |        2 |     96% |  111, 116 |
| p2pfl/learning/compression/quantization\_strategy.py                          |      223 |       72 |     68% |89-90, 95, 98, 166, 169, 189, 192, 197, 203, 237-244, 248, 314, 317, 322-323, 328-337, 344, 364-366, 384-415, 445, 448, 451, 455, 490, 493, 496, 499, 502, 506, 510, 514, 517, 522, 540, 543, 546, 554 |
| p2pfl/learning/compression/topk\_strategy.py                                  |       28 |        1 |     96% |        82 |
| p2pfl/learning/compression/zlib\_strategy.py                                  |        7 |        0 |    100% |           |
| p2pfl/learning/dataset/\_\_init\_\_.py                                        |        0 |        0 |    100% |           |
| p2pfl/learning/dataset/p2pfl\_dataset.py                                      |       74 |       16 |     78% |53, 140, 154, 177, 191, 196, 280-281, 296-297, 312-313, 327-328, 358-359 |
| p2pfl/learning/dataset/partition\_strategies.py                               |      104 |       12 |     88% |59, 140, 191-196, 198, 269, 420, 422, 424 |
| p2pfl/learning/frameworks/\_\_init\_\_.py                                     |        5 |        0 |    100% |           |
| p2pfl/learning/frameworks/callback.py                                         |       13 |        1 |     92% |        42 |
| p2pfl/learning/frameworks/callback\_factory.py                                |       36 |       13 |     64% |53, 71-82, 93-94, 100-101 |
| p2pfl/learning/frameworks/exceptions.py                                       |        4 |        0 |    100% |           |
| p2pfl/learning/frameworks/flax/\_\_init\_\_.py                                |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/flax/flax\_dataset.py                               |       17 |        9 |     47% |     55-69 |
| p2pfl/learning/frameworks/flax/flax\_learner.py                               |       86 |       86 |      0% |    21-181 |
| p2pfl/learning/frameworks/flax/flax\_model.py                                 |       54 |       34 |     37% |53-58, 62, 66-69, 79, 92-101, 111-123, 133-140, 153-154, 164 |
| p2pfl/learning/frameworks/learner.py                                          |       74 |        9 |     88% |54, 59, 62, 91, 115, 161, 166, 177, 188 |
| p2pfl/learning/frameworks/learner\_factory.py                                 |       20 |        5 |     75% |     50-56 |
| p2pfl/learning/frameworks/p2pfl\_model.py                                     |       58 |       10 |     83% |64, 68, 99-100, 110, 123, 147, 164, 170, 194 |
| p2pfl/learning/frameworks/pytorch/\_\_init\_\_.py                             |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/pytorch/callbacks/\_\_init\_\_.py                   |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/pytorch/callbacks/scaffold\_callback.py             |       62 |       45 |     27% |41-47, 52, 63-74, 87-88, 103-110, 121-140, 143, 147-150 |
| p2pfl/learning/frameworks/pytorch/lightning\_dataset.py                       |       26 |        2 |     92% |  101, 103 |
| p2pfl/learning/frameworks/pytorch/lightning\_learner.py                       |       74 |       14 |     81% |78, 82, 88, 112-118, 122-124, 146-152 |
| p2pfl/learning/frameworks/pytorch/lightning\_logger.py                        |       22 |        2 |     91% |    45, 54 |
| p2pfl/learning/frameworks/pytorch/lightning\_model.py                         |       27 |        2 |     93% |     98-99 |
| p2pfl/learning/frameworks/simulation/\_\_init\_\_.py                          |        7 |        0 |    100% |           |
| p2pfl/learning/frameworks/simulation/actor\_pool.py                           |      137 |       25 |     82% |45-46, 50-56, 60-66, 117, 138, 238-239, 256-260, 324, 329, 338, 355-356 |
| p2pfl/learning/frameworks/simulation/utils.py                                 |       29 |        6 |     79% |44-45, 76, 84, 88-94 |
| p2pfl/learning/frameworks/simulation/virtual\_learner.py                      |       52 |       10 |     81% |107, 111, 123-125, 130, 147-149, 153 |
| p2pfl/learning/frameworks/tensorflow/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/tensorflow/callbacks/\_\_init\_\_.py                |        0 |        0 |    100% |           |
| p2pfl/learning/frameworks/tensorflow/callbacks/keras\_logger.py               |       20 |        0 |    100% |           |
| p2pfl/learning/frameworks/tensorflow/callbacks/scaffold\_callback.py          |       69 |       51 |     26% |44-47, 52-59, 63, 78-84, 89, 93-116, 132, 142-159, 163 |
| p2pfl/learning/frameworks/tensorflow/keras\_dataset.py                        |       17 |        2 |     88% |    63, 65 |
| p2pfl/learning/frameworks/tensorflow/keras\_learner.py                        |       63 |       11 |     83% |81, 85, 107-109, 115, 124, 130-133 |
| p2pfl/learning/frameworks/tensorflow/keras\_model.py                          |       25 |        1 |     96% |        69 |
| p2pfl/management/\_\_init\_\_.py                                              |        0 |        0 |    100% |           |
| p2pfl/management/cli.py                                                       |       59 |       59 |      0% |    21-175 |
| p2pfl/management/launch\_from\_yaml.py                                        |      149 |      149 |      0% |    20-282 |
| p2pfl/management/logger/\_\_init\_\_.py                                       |       14 |        1 |     93% |        42 |
| p2pfl/management/logger/decorators/async\_logger.py                           |       22 |       12 |     45% |34-46, 51, 55-58 |
| p2pfl/management/logger/decorators/file\_logger.py                            |       38 |       29 |     24% |35-38, 42-89 |
| p2pfl/management/logger/decorators/logger\_decorator.py                       |       43 |       18 |     58% |45, 56, 60, 70, 80, 93, 105, 119, 133, 147, 157, 167, 178, 189, 199, 209, 219, 232 |
| p2pfl/management/logger/decorators/ray\_logger.py                             |       53 |       11 |     79% |67-69, 80, 84, 158, 213, 224, 234, 244, 257 |
| p2pfl/management/logger/decorators/singleton\_logger.py                       |        4 |        0 |    100% |           |
| p2pfl/management/logger/decorators/web\_logger.py                             |       60 |       40 |     33% |48-56, 70-72, 83-85, 98-99, 110-113, 127-140, 153-155, 165-172, 182-193 |
| p2pfl/management/logger/logger.py                                             |      125 |       79 |     37% |72-83, 103-125, 136, 141-146, 160-163, 173, 186, 241, 254-265, 284-308, 322, 336, 351-355, 366-371, 386, 397-401, 411-412, 422, 432, 445 |
| p2pfl/management/metric\_storage.py                                           |       56 |       36 |     36% |52-53, 77-100, 110, 123, 137, 152, 177-178, 193-214, 224, 237, 251 |
| p2pfl/management/node\_monitor.py                                             |       38 |       26 |     32% |43-52, 56, 60-65, 69-82, 86 |
| p2pfl/management/p2pfl\_web\_services.py                                      |       75 |       60 |     20% |53-55, 70-75, 79-81, 93-105, 115, 129-152, 168-193, 208-232, 246-266, 270 |
| p2pfl/node.py                                                                 |      135 |       35 |     74% |185-188, 212, 235-236, 256-257, 274-276, 289-291, 305-307, 331, 363, 382-383, 387-393, 413-415, 418-428 |
| p2pfl/node\_state.py                                                          |       43 |        3 |     93% |89, 113, 125 |
| p2pfl/settings.py                                                             |       95 |        9 |     91% |   158-169 |
| p2pfl/stages/\_\_init\_\_.py                                                  |        0 |        0 |    100% |           |
| p2pfl/stages/base\_node/\_\_init\_\_.py                                       |        0 |        0 |    100% |           |
| p2pfl/stages/base\_node/gossip\_model\_stage.py                               |       37 |        3 |     92% |50, 64, 77 |
| p2pfl/stages/base\_node/round\_finished\_stage.py                             |       37 |        2 |     95% |    51, 63 |
| p2pfl/stages/base\_node/start\_learning\_stage.py                             |       49 |        2 |     96% |   64, 109 |
| p2pfl/stages/base\_node/train\_stage.py                                       |       77 |        4 |     95% |53, 100-101, 161 |
| p2pfl/stages/base\_node/vote\_train\_set\_stage.py                            |       87 |        6 |     93% |52, 75-76, 143-144, 185 |
| p2pfl/stages/base\_node/wait\_agg\_models\_stage.py                           |       25 |        2 |     92% |    45, 57 |
| p2pfl/stages/stage.py                                                         |       19 |        6 |     68% |32, 37, 62-65 |
| p2pfl/stages/stage\_factory.py                                                |       24 |        1 |     96% |        59 |
| p2pfl/stages/workflows.py                                                     |       26 |        1 |     96% |        52 |
| p2pfl/utils/check\_ray.py                                                     |       13 |        2 |     85% |    30, 48 |
| p2pfl/utils/node\_component.py                                                |       31 |        0 |    100% |           |
| p2pfl/utils/seed.py                                                           |       34 |       12 |     65% |58, 60-61, 64-72, 80-81 |
| p2pfl/utils/singleton.py                                                      |        7 |        0 |    100% |           |
| p2pfl/utils/topologies.py                                                     |       44 |        3 |     93% | 72, 94-95 |
| p2pfl/utils/utils.py                                                          |       59 |        6 |     90% |94, 101, 113-114, 133, 156 |
|                                                                     **TOTAL** | **4564** | **1348** | **70%** |           |


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