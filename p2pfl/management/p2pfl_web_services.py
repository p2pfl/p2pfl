import requests


class P2pflWebServices:
    def __init__(self, url: str, key: str) -> None:
        self.__url = url
        # http warning
        if not url.startswith("https://"):
            print(
                "P2pflWebServices Warning: Connection must be over https, traffic will not be encrypted"
            )
        self.__key = key

    def send_log(self, time: str, node: str, level: int, message: str):
        data = {"time": time, "node": node, "level": level, "message": message}
        headers = {"Content-Type": "application/json"}
        headers["Authorization"] = f"Bearer {self.__key}"
        try:
            response = requests.post(
                self.__url + "/log", json=data, headers=headers, timeout=5
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(node, f"Error logging message: {e}")
            raise e

    def send_metric(self, node: str, metric: str, value: float):
        """
        TODO:
            - kpis
            - model logs
        """
        raise NotImplementedError

    def get_pending_actions(self):
        raise NotImplementedError
