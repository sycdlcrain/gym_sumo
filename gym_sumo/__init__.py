import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='sumo-v0',
    entry_point='gym_sumo.envs:SumoEnv',
    timestep_limit=100,
)

