import numpy as np
from car_env import DrivingGridEnv

def test_env():
    # Instantiate (you can pass render_mode="human" to see ASCII)
    env = DrivingGridEnv(render_mode="human")

    # 1) Check spaces
    print("Obs space:", env.observation_space)
    print("Action space:", env.action_space)

    # 2) Reset
    obs, info = env.reset()
    print("Initial obs shape:", obs.shape)
    env.render()  # print the grid

    # 3) Step through a few random actions
    for t in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {t:02}, action={action}, reward={reward:.2f}, "
              f"term={terminated}, trunc={truncated}")
        env.render()
        if terminated or truncated:
            print("Episode ended, resetting…\n")
            obs, info = env.reset()
            env.render()

    env.close()

if __name__ == "__main__":
    test_env()