from cartpole import CartPoleEnv
import gym
import numpy as np

if __name__ == "__main__":
    cart = CartPoleEnv()
    cart.reset()
    # while True:
    #     cart.render()
    #     cart.step(0)
    #     cart.render()
    #     cart.step(1)
    a = cart.step(1)
    print(a)
    cart.close()
    env = gym.make("CartPole-v1")
# print(env.action_space.n)
#     # print(env.action_space.n)
#     # Observation = [30,30,50,50]

#     # q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n]))
#     # print(q_table.shape)


