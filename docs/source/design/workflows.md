# 🔄 Workflows
🚧 Workflows

[TODO]



🌟 Ready? **You can view next**: > [Stages & Commands](docs-stages-commands.md)

<div style="position: fixed; bottom: 10px; right: 10px; font-size: 0.9em; padding: 10px; border: 1px solid #ccc; border-radius: 5px;"> 🌟 You Can View Next: <a href="docs-stages-commands.md">Stages & Commands</a> </div>





## Main workflow

The main workflow of the library is as follows:

> TENGO QUE ACTUALZARLO, CAMBIADO | AGREGAR DESCRIPCIONES DE CADA ETAPA

```{eval-rst}
.. mermaid::
    :align: center

    graph LR
        A(StartLearningStage) --> B(VoteTrainSetStage)
        B -- Node in trainset? --> C(TrainStage)
        B -- Node not in trainset? --> D(WaitAggregatedModelsStage)
        C --> E(GossipModelStage) 
        D --> E
        E --> F(RoundFinishedStage)
        F -- No more rounds? --> Finished
        F -- More rounds? --> B
```