from gym.envs.registration import register

register(
    id='HungryGeese-v0',
    entry_point='gym_hungry_geese.envs:HungryGeeseEnv',
)