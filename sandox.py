import gym

env = gym.make("ALE/Breakout-v5", frameskip=1, render_mode="human")
env = gym.wrappers.AtariPreprocessing(env)
env = gym.wrappers.FrameStack(env, 4)
env.reset()
for _ in range(10000):
    observation, reward, done, info = env.step(env.action_space.sample())

    print(observation)
    print(observation.shape)
