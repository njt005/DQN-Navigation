import typing as t
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Network


class Agent:
    """
    Agent that utilizes a linear MLP Q-Network to approximate
    state-action value in environment
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: t.Tuple[int, ...],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 4,
        update_every: int = 100_000,
        tau: float = 1e-3,
        seed: int = 5,
    ):
        """__init__

        Initialize Agent with DQN structure.

        Parameters
        ----------
        state_size : int
            size of input state
        action_size : int
            size of output (number of available actions)
        hidden_size : t.Tuple[int, ...]
            hidden layer sizes
        learning_rate : float
            learning rate for updating weights
        gamma : float
            reward discount
        batch_size : int
            size of batches used for training
        update_every : int
            update target network this often
        tau : float
            soft update parameter
        seed : int, optional
            random seed for network, by default 5
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.update_every = update_every
        self.tau = tau
        self.seed = seed

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q Networks
        self.qnetwork_local = Network(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_sizes=self.hidden_sizes,
            seed=self.seed,
        ).to(self.device)

        self.qnetwork_target = Network(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_sizes=self.hidden_sizes,
            seed=self.seed,
        ).to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(
            action_size=action_size,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=seed,
        )
        # Initialize time step
        self.t_step = 0

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """step

        Add state, action, reward, next_state, and done to memory
        Update network weights if step is an update_every step

        Parameters
        ----------
        state : np.ndarray
            state of agent
        action : int
            action agent took
        reward : int
            reward received from action
        next_state : np.ndarray
            next state of agent
        done : bool
            end of episode
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(batch_size=self.batch_size)
                self.learn(experiences=experiences, gamma=self.gamma, tau=self.tau)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        """act

        Pre-process state and get action given current state based on epsilon-greedy policy.

        Parameters
        ----------
        state : np.ndarray
            current state of agent
        epsilon : float
            parameter for exploration vs. exploitation

        Returns
        -------
        int
            action for agent to take
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if np.random.uniform(0, 1, 1) > epsilon:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return int(np.random.choice(np.arange(self.action_size)))

    def learn(
        self, experiences: t.Tuple[torch.Tensor], gamma: float, tau: float
    ) -> None:
        """learn

        Update value parameters using given batch of experience tuples.

        Parameters
        ----------
        experiences : t.Tuple[torch.Tensor]
            tuple of (s, a, r, s', done) tuples
        gamma : float
           reward discount factor
        tau : float
            soft update parameter
        """
        states, actions, rewards, next_states, dones = experiences
        Q_targets_ = self.qnetwork_target(next_states)

        # Get max predicted Q values (for next states) from target model
        Q_targets_ = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_ * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

    def soft_update(self, local_model: Network, target_model: Network, tau: float):
        """soft_update

        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter

        Parameters
        ----------
        local_model : Network
            weights copied from local model to target
        target_model : Network
            target model that receives copies of weights from local network
        tau : float
            soft update parameter
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self, action_size: int, buffer_size: int, batch_size: int, seed: int = 5
    ):
        """__init__

        Initialize a replay buffer object

        Parameters
        ----------
        action_size : int
            size of action space
        buffer_size : int
            size of replay buffer to sample from
        batch_size : int
            size of batches
        seed : int, optional
            random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = random.seed(seed)

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """add

        Add experience to memory

        Parameters
        ----------
        state : np.ndarray
            state of agent
        action : int
            action agent took
        reward : int
            reward received from action
        next_state : np.ndarray
            next state of agent
        done : bool
            end of episode
        """
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, batch_size: int) -> t.Tuple[torch.Tensor]:
        """sample

        Sample from memory to the length of the batch_size

        Parameters
        ----------
        batch_size : int
            size of batches

        Returns
        -------
        t.Tuple[float.Tensor]
            tuple of sampled (s, a, r, s', done)
        """
        sampled_idx = np.random.choice(np.arange(len(self.memory)), batch_size)
        experiences = [self.memory[idx] for idx in sampled_idx]
        experiences = [
            experience for experience in experiences if experience is not None
        ]

        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences]))
            .float()
            .to(self.device)
        )
        actions = (
            torch.from_numpy(np.vstack([e.action for e in experiences]))
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e.reward for e in experiences]))
            .float()
            .to(self.device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e.next_state for e in experiences]))
            .float()
            .to(self.device)
        )
        dones = (
            torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8))
            .float()
            .to(self.device)
        )

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
