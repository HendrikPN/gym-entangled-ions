import sys,os
import numpy as np
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'gym_entangled_ions/envs'))
from entangled_ions_env import EntangledIonsEnv

params = {'num_ions': 3, "dim": 3}
env = EntangledIonsEnv(**params)


env.reset()

o,r,_,_=env.step(1)

o,r,_,_=env.step(1)

o,r,_,_=env.step(3)

# print(o)

# print(o.real)
# print(o.imag)

# print(o.shape)

# print(o.dtype)

# print(env.observation_space)

print(env.action_labels)

# timing = []
# timing_a = []
# for e in range(100):
#     env.reset()
#     for steps in range(20):
#         a = np.random.choice(env.num_actions)
#         o,r,_ = env.step(a)
#         t1 = time.time()
#         srv=env.srv(o)
#         t2 = time.time()
#         t3 = time.time()
#         srv_a=env.srv_alt(o)
#         t4 = time.time()
#         timing.append(t2-t1)
#         timing_a.append(t4-t3)
#         assert srv == srv_a

# t = sum(timing)/len(timing)
# t_a = sum(timing_a)/len(timing_a)

# print(t)
# print(t_a)

if __name__ == "__main__":
    import gym
    name = 'entangled-ions-v0'
    # dimension of the qudit
    DIM = 3
    # number of qudits
    NUM_IONS = 5
    # number of operations:
    MAX_OP = 10
    # conditioons for the reward function
    goal_srv_max = 3
    env = gym.make(name, dim=DIM, num_ions=NUM_IONS, max_op=MAX_OP, goal_srv_max=goal_srv_max)
    print(env.action_labels)
    done = False
    while not done:
        action = np.random.choice(env.num_actions)
        o, r, done, _ = env.step(action)
        print(o)
        print(f'reward: {r}')