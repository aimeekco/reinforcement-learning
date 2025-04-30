import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DrivingGridEnv(gym.Env):
    """
    endless-runner style driving grid:
    - car is fixed on bottom row, walls in cols 0 & 6, drivable in cols 1–5.
    - obstacles spawn at the top and scroll downward each step.
    - action: discrete left/straight/right + 5 throttle levels mapped to ∈ [0,1].
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        grid_shape=(10, 7),
        max_steps=500,
        throttle_levels=5,
        obstacle_spawn_rate=0.2,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.rows, self.cols = grid_shape
        self.max_steps = max_steps
        self.obstacle_spawn_rate = obstacle_spawn_rate
        self.throttle_levels = throttle_levels
        self.safe_col = None

        # 0=empty, 1=wall, 2=car, 3=obstacle
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(self.rows, self.cols),
            dtype=np.int8
        )

        # hybrid action: (steer ∈ {L,S,R}, throttle ∈ [0,1])
        steering = spaces.Discrete(3)  # 0=left, 1=straight, 2=right
        throttle = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3 * self.throttle_levels)

        # internal state
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.car_col = None
        self.obstacles = []
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.obstacles.clear()
        self.safe_col = self.cols // 2

        # build walls & empty grid
        self.grid[:] = 0
        self.grid[:, 0] = self.grid[:, -1] = 1

        # place car at bottom-center
        self.car_col = self.cols // 2
        self.grid[-1, self.car_col] = 2
        
        self.safe_col = self.cols // 2

        return self.grid.copy(), {}

    def spawn_obstacles(self):
        drivable = list(range(1, self.cols - 1))
        new_row = [0] * self.cols
        for c in drivable:
            if self.np_random.random() < self.obstacle_spawn_rate:
                new_row[c] = 3
        filled = [c for c in drivable if new_row[c] == 3]
        if len(filled) == len(drivable):
            drop = self.np_random.choice(filled)
            new_row[drop] = 0
        for c in drivable:
            if new_row[c] == 3:
                self.obstacles.append((0, c))

    def step(self, action):
        steering = action // self.throttle_levels
        thr_idx  = action % self.throttle_levels

        # move car
        if steering == 0:
            self.car_col = max(1, self.car_col - 1)
        elif steering == 2:
            self.car_col = min(self.cols - 2, self.car_col + 1)

        # compute speed & scroll obstacles
        speed       = thr_idx / (self.throttle_levels - 1)
        self.obstacles = [(r + 1, c) for (r, c) in self.obstacles]

        # random safe-column shift & new obstacles
        move = self.np_random.choice([-1, 0, 1])
        self.safe_col = np.clip(self.safe_col + move, 1, self.cols-2)
        for c in range(1, self.cols-1):
            if c != self.safe_col and self.np_random.random() < self.obstacle_spawn_rate:
                self.obstacles.append((0, c))

        # rebuild grid
        self.grid[:]       = 0
        self.grid[:, 0]    = self.grid[:, -1] = 1
        for (r, c) in self.obstacles:
            if 0 <= r < self.rows:
                self.grid[r, c] = 3
        self.grid[-1, self.car_col] = 2

        # check for collision & reward  
        terminated = any(r == self.rows-1 and c == self.car_col
                         for r, c in self.obstacles)
        reward     = speed - (10.0 if terminated else 0.0)

        # truncation
        self.step_count += 1
        truncated = (self.step_count >= self.max_steps)

        # pack info and return
        info = {"distance": self.step_count, "crash": terminated}
        obs  = self.grid.copy()
        return obs, reward, terminated, truncated, info


    def render(self, mode="human"):
        if mode == "human":
            #ASCII‐visualization
            chars = {0: " . ", 1: "|", 2: " 🚘", 3: " 🚧"}
            print("\n".join("".join(chars[v] for v in row)
                             for row in self.grid))
            print(f"Step {self.step_count}")
        elif mode == "rgb_array":
            # return an (rows×cols×3) uint8 image
            frame = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
            frame[self.grid == 1] = [100, 100, 100]
            frame[self.grid == 2] = [0, 200, 0]
            frame[self.grid == 3] = [200, 0, 0]
            return frame

    def close(self):
        pass
