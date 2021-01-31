from cartpole import CartPoleEnv

if __name__ == "__main__":
    cart = CartPoleEnv()
    cart.reset()
    while True:
        cart.render()
        cart.step(0)
        cart.render()
        cart.step(1)
    cart.close()

