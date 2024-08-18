import asyncio
import ray

# We set num_cpus to zero because this actor will mostly just block on I/O.
@ray.remote(num_cpus=0)
class SignalActor:
    def __init__(self):
        self.ready_event = asyncio.Event()

    def send(self, clear=False):
        self.ready_event.set()
        if clear:
            self.ready_event.clear()

    async def wait(self, should_wait=True):
        if should_wait:
            await self.ready_event.wait()


@ray.remote
def wait_and_go(signal):
    ray.get(signal.wait.remote())