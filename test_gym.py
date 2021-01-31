from cartpole import CartPoleEnv

def choose_action(state):
    action = 0
    if state[2] > 0:
        action = 0
    else:
        action = 1
    return action

if __name__ == "__main__":
    cart = CartPoleEnv()
    cart.reset()
    action = 0
    while True:
        cart.render()
        state, reward, end, thing = cart.step(action)
        print(state)
        if end:
            cart.reset()
        else:
            action = choose_action(state)
    cart.close()

