import os
import numpy as np
import torch
from unityagents import UnityEnvironment

from model import Network
from train_agent import N_LAGS, HIDDEN_SIZES

# Agent
# Environment - update based on Unity environment location
# Put folder in same directory as this file
ENV_PATH = "Banana_Windows_x86_64/Banana.exe"
SAVED_MODEL = "models/q_final_n333.pt"


def main():
    # Initialize environment and get env info to initialize Agent
    base_path = os.path.dirname(os.path.abspath(__file__))
    env_full_path = os.path.join(base_path, ENV_PATH)
    env = UnityEnvironment(file_name=env_full_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    state_size = len(env_info.vector_observations[0])
    if N_LAGS is not None:
        state_size = state_size * (N_LAGS + 1)
    action_size = brain.vector_action_space_size

    # Get trained agent
    q_net = Network(
        state_size=state_size, action_size=action_size, hidden_sizes=HIDDEN_SIZES
    )
    model_path = os.path.join(base_path, SAVED_MODEL)
    q_net.load_state_dict(torch.load(model_path))
    q_net.eval()

    # Watch single episode
    state = env_info.vector_observations[0]
    score = 0
    states = []
    while True:
        states.append(state)
        if N_LAGS is not None:
            if len(states) <= N_LAGS:
                for _ in range(N_LAGS + 1):
                    states.append(state)
            state = np.concatenate(states[-N_LAGS - 1 :])
        action_values = q_net(torch.FloatTensor(state))
        action = int(np.argmax(action_values.cpu().data.numpy()))
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:
            break

    print(f"Single episode score: {score}")


if __name__ == "__main__":
    main()
