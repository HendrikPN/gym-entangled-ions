from gym.envs.registration import register

register(
    id='entangled-ions-v0',
    entry_point='gym_entangled_ions.envs:EntangledIonsEnv',
)
