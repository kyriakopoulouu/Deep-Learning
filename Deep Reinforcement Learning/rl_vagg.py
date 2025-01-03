import os
import gymnasium as gym
import tensorflow as tf
import imageio
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback,StopTrainingOnNoModelImprovement
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

   # Neural network for predicting action values
class CustomCNN(BaseFeaturesExtractor):
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int=128):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels,128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(409600,128),
            nn.Linear(128,64)
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(64, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128)
)


if __name__ == "__main__":
    # Create the base environment
    base_env = make_atari_env("ALE/MarioBros-v5", n_envs=6, seed=42,vec_env_cls=SubprocVecEnv)

    # Frame-stacking with 4 frames
    train_env = VecFrameStack(base_env, n_stack=4)

    # Separate evaluation env with the same base environment
    eval_env = VecFrameStack(base_env, n_stack=4)



    eval_callback = EvalCallback(eval_env, 
                                best_model_save_path="./logs/best_model", 
                                log_path="./logs/results", 
                                eval_freq=1000, 
                                verbose=1
    )
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/")
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])


    CHECKPOINT_DIR = 'train'
    LOG_DIR = 'logs'
 
    # Initialize agent
    model = PPO('CnnPolicy', 
        train_env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        policy_kwargs=policy_kwargs, 
        learning_rate=0.01,
        seed =42
    ) 
    model.learn(total_timesteps=200000,progress_bar=True, callback=[callback])

    train_env.metadata['render_fps'] = 29

    # Create gif
    images = []
    obs = model.env.reset()
    img = model.env.render(mode="rgb_array")

    for i in range(300):
        images.append(img)
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = model.env.step(action)
        img = model.env.render(mode="rgb_array")

    # Save the GIF
    imageio.mimsave("Cnnpolicy_ORIGINAL.gif", [np.array(img) for i, img in enumerate(images) if i % 2 == 0], duration=500)