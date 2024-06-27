"""
explicar que esto gestion el workflow del hilo principal de cada nodo
"""


def StageWokflow():

    def __init__(self, stage_list):
        self.__stage_list = stage_list

    def run(self):
        while True:
            for stage in self.__stage_list:
                exit = stage.run()
                if exit:
                    break
