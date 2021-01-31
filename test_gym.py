from cartpole import CartPoleEnv

if __name__ == "__main__":
    cart = CartPoleEnv()
    while True:
        cart.render()
        cart.step(1)
