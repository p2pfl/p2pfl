from p2pfl.management.logger.simple_logger import SimpleP2PFLogger

# Check if 'ray' is installed in the Python environment
import importlib.util
ray_installed = importlib.util.find_spec("ray") is not None

# Create the logger depending on the availability of 'ray'
if ray_installed:
    from p2pfl.management.logger.ray_logger import RayP2PFLogger

    # Logger actor singleton
    logger = RayP2PFLogger(SimpleP2PFLogger())
else:
    from p2pfl.management.logger.async_logger import AsyncLocalLogger
    # Logger actor singleton
    logger = AsyncLocalLogger(SimpleP2PFLogger())
