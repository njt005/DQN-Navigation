import os
import typing as t
import numpy as np
import torch
from collections import deque
from unityagents import UnityEnvironment

from agent import Agent
from plotting import create_reward_plot

# Agent parameters
BUFFER_SIZE = 100_000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LEARNING_RATE = 1e-3
UPDATE_EVERY = 4
HIDDEN_SIZES = (128, 64)
SEED = 1

# Environment - update based on Unity environment location
# Put folder in same directory as this file
ENV_PATH = "Banana_Windows_x86_64/Banana.exe"

# Training parameters
N_EPISODES = 1_000
REWARD_GOAL = 13
MAX_ITER_EPISODE = 10_000
EPSILON = 1.0
EPSILON_DECAY = 1e-2
EPSILON_MIN = 1e-2
N_LAGS = 4
PLOT_REWARD = True


def train_agent(
    agent: Agent,
    env: UnityEnvironment,
    brain_name: str,
    n_episodes: int,
    epsilon: float,
    epsilon_decay: float,
    epsilon_min: float,
    max_iter_episode: int,
    reward_goal: int,
    n_lags: t.Optional[int],
    plot_reward: bool = True,
    save_path: t.Optional[str] = None,
):
    """train_agent

    Takes a DQN agent and environment as input and trains the agent.

    Parameters
    ----------
    agent : Agent
        agent
    env : UnityEnvironment
        Unity environment
    brain_name : str
        brain name
    epsilon : float
        exploration / exploitation tradeoff parameter
    epsilon_decay : float
        rate to decay epsilon
    epsilon_min : float
        min espsilon
    n_episodes : int
        number of episodes to train
    max_iter_episode : int
        maximum number of iterations in an episode
    reward_goal : int
        reward goal to stop training
    n_lags : t.Optional[int]
        number of lags to lag current state
        if None - no lags
    plot_reward : bool
        plot reward once finished training
    save_path : t.Optional[str]
        path to save training results
    """
    scores = []
    scores_window = deque(maxlen=100)
    for episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = []
        state = env_info.vector_observations[0]
        score = 0

        # Update epsilon based on decay rate
        if epsilon_decay is not None:
            epsilon = np.max([epsilon - epsilon * epsilon_decay, epsilon_min])

        for _ in range(max_iter_episode):
            # Store current state and add lags
            states.append(state)
            if n_lags is not None:
                if len(states) <= n_lags:
                    for _ in range(n_lags + 1):
                        states.append(state)

                state = np.concatenate(states[-n_lags - 1 :])

            # Take action and update networks
            action = agent.act(state=state, epsilon=epsilon)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            if n_lags is not None:
                next_state_save = np.concatenate(states[-n_lags:] + [next_state])
            else:
                next_state_save = next_state
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state_save, done)
            score += reward
            state = next_state

            if done:
                break

        # Append scores
        scores_window.append(score)
        scores.append(score)

        print(
            "\rEpisode {}\tAverage Score: {:.2f}, Epsilon: {:.4f}".format(
                episode, np.mean(scores_window), epsilon
            ),
            end="",
        )
        if episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    episode, np.mean(scores_window)
                )
            )
        if np.mean(scores_window) >= reward_goal:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    episode - 100, np.mean(scores_window)
                )
            )
            if save_path is not None:
                model_save_path = os.path.join(save_path, "models")
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)

                torch.save(
                    agent.qnetwork_local.state_dict(),
                    os.path.join(model_save_path, f"q_final_n{episode-100}.pt"),
                )
                env.close()
            break

    if plot_reward:
        if save_path is not None:
            figure_save_path = os.path.join(save_path, "figures")
            if not os.path.exists(figure_save_path):
                os.makedirs(figure_save_path)
        create_reward_plot(
            scores=scores,
            smooth_n=100,
            save_path=os.path.join(figure_save_path, f"scores_n{episode-100}.png"),
        )


def main():
    # Initialize environment and get env info to initialize Agent
    base_path = os.path.dirname(os.path.abspath(__file__))
    env_full_path = os.path.join(base_path, ENV_PATH)
    env = UnityEnvironment(file_name=env_full_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    if N_LAGS is not None:
        state_size = state_size * (N_LAGS + 1)
    action_size = brain.vector_action_space_size

    # Initialize agent
    agent = Agent(
        state_size=state_size,
        action_size=action_size,
        hidden_sizes=HIDDEN_SIZES,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        update_every=UPDATE_EVERY,
        tau=TAU,
        seed=SEED,
    )

    train_agent(
        agent=agent,
        env=env,
        brain_name=brain_name,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
        n_episodes=N_EPISODES,
        max_iter_episode=MAX_ITER_EPISODE,
        reward_goal=REWARD_GOAL,
        n_lags=N_LAGS,
        plot_reward=PLOT_REWARD,
        save_path=base_path,
    )


if __name__ == "__main__":
    main()
