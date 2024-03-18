"""
import requests
from p2pfl.logging.logger import logger


class P2pflWebServices:
    def __init__(self, url: str, key: str) -> None:
        self.__url = url
        # http warning
        if not url.startswith("http://"):
            logger.warning(
                "P2pflWebServices",
                "Warning: URL should start with http://, traffic will not be encrypted",
            )
        self.__key = key

    def send_log(self, node: str, level: int, message: str):
        data = {"node": node, "level": level, "message": message}
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {self.__key}"
        try:
            response = requests.post(
                self.__url + "", json=data, headers=headers, timeout=5
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(node, f"Error logging message: {e}")
            raise e

    def send_metric(self, node: str, metric: str, value: float):
        data = {"node": node, "metric": metric, "value": value}
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {self.__key}"
        try:
            response = requests.post(
                self.__url + "", headers=headers, data=data, timeout=5
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(node, f"Error logging message: {e}")
            raise e
"""

"""
TODO:
    - kpis
    - remote commands
"""
