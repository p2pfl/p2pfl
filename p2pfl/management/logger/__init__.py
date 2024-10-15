from p2pfl.management.logger.ray_logger import RayP2PFLogger
from p2pfl.management.logger.simple_logger import SimpleP2PFLogger

# Logger actor singleton
logger = RayP2PFLogger(SimpleP2PFLogger())