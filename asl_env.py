import os
import cv2
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete

class ASLEnv(Env):
    def __init__(self, data_path, labels, img_size=(16, 16), seed=None, episode_limit=20):
        super(ASLEnv, self).__init__()
        self.seed = seed
        # Environment parameters
        self.data_path = data_path
        self.labels = labels  # List of class labels, e.g., ["A", "B", "C", ..., "Z", "0", ..., "9"]
        self.img_size = img_size
        self.current_index = 0
        
        self.episode_limit = episode_limit

        # Loading the dataset
        self.data, self.targets = self.load_data()
        self.num_classes = len(self.labels)
        
        # Define action and observation spaces
        self.action_space = Discrete(self.num_classes)  # One action per class (A-Z, 0-9)
        self.observation_space = Box(low=0, high=255, shape=(img_size[0], img_size[1], 3), dtype=np.uint8)

    def get_env_info(self):
        """Returns environment info such as action and observation space details."""
        return {
            "action_space": self.action_space.n,  # Number of possible actions
            "observation_space": self.observation_space.shape,  # Shape of the observation space
            "episode_limit": self.episode_limit,  # Maximum number of steps in an episode
            "n_agents": 1,  # Single agent in this environment
            "n_actions": self.num_classes,  # Number of possible actions (A-Z, 0-9)
            "state_shape": self.observation_space.shape,
            "obs_shape": self.observation_space.shape
        }

    def load_data(self):
        """Load the dataset from the given path."""
        data = []
        targets = []
        for label in self.labels:
            label_path = os.path.join(self.data_path, label)
            for img_file in os.listdir(label_path):
                img = cv2.imread(os.path.join(label_path, img_file))
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    data.append(img)
                    targets.append(self.labels.index(label))  # Encode label as an integer
        return np.array(data), np.array(targets)

    def reset(self):
        """Reset the environment to the first image."""
        self.current_index = 0
        return self.data[self.current_index]

    def step(self, action):
        """Step function to take an action, calculate reward and return the next state."""
        # Calculate reward (1 if action is correct, -1 otherwise)
        reward = 1 if action == self.targets[self.current_index] else -1
        
        # Move to the next image
        self.current_index = (self.current_index + 1) % len(self.data)
        done = self.current_index == 0  # Episode ends after a full loop through the data
        
        return self.data[self.current_index], reward, done, {}

    def get_state(self):
        """Returns the current observation as the state."""
        return self.data[self.current_index]

    def get_avail_actions(self):
        """Returns available actions for the agent. All actions are available."""
        return np.ones(self.num_classes)

    def get_obs(self):
        """Returns the current observation for the agent."""
        return self.data[self.current_index]

    def get_agent_obs(self, agent_idx):
        """Returns the observation for a specific agent (if needed for multi-agent environments)."""
        # In your case, there's only one agent, so return the same observation.
        return self.data[self.current_index]
