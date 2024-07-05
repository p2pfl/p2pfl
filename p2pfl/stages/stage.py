from typing import Union


class Stage:

    @staticmethod
    def name():
        raise NotImplementedError("Stage name not implemented.")

    @staticmethod
    def execute() -> Union["Stage", None]:
        raise NotImplementedError("Stage execute not implemented.")


class StageWokflow:
    def __init__(self, first_stage: Stage, early_stopping_fn=lambda: False):
        self.current_stage = first_stage
        self.early_stopping_fn = early_stopping_fn

    def run(self, context):
        while True:
            self.current_stage = self.current_stage.execute(context)
            if self.current_stage is None or self.early_stopping_fn():
                break
