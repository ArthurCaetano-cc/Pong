import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import cv2
import ale_py

# ⚡ Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
MEMORY_SIZE = 100000
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.02
EXPLORATION_DECAY = 0.995
TARGET_UPDATE = 1000  # Update target network every N steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🎮 Preprocessing function
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (84, 84))  # Resize to 84x84
    return frame / 255.0  # Normalize pixel values

# 🎮 Environment Setup
class PongEnv:
    def __init__(self):
        self.env = gym.make("ALE/Pong-v5", frameskip=4)
        self.state_buffer = collections.deque(maxlen=4)  # Store last 4 frames
    
    def reset(self):
        state, _ = self.env.reset()
        processed = preprocess(state)
        for _ in range(4):  # Fill buffer with initial frame
            self.state_buffer.append(processed)
        return np.array(self.state_buffer, dtype=np.float32)
    
    def step(self, action):
        next_state, reward, done, _, _ = self.env.step(action)
        processed = preprocess(next_state)
        self.state_buffer.append(processed)
        return np.array(self.state_buffer, dtype=np.float32), reward, done

# 🧠 Deep Q-Network (CNN)
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 🎮 Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32, device=DEVICE),
            torch.tensor(actions, dtype=torch.int64, device=DEVICE).unsqueeze(-1),
            torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
            torch.tensor(next_states, dtype=torch.float32, device=DEVICE),
            torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)

# 🎯 DQN Agent
class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.dqn = DQN(input_shape, num_actions).to(DEVICE)
        self.target_dqn = DQN(input_shape, num_actions).to(DEVICE)
        self.target_dqn.load_state_dict(self.dqn.state_dict())  # Sync target network
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE)
        self.exploration_rate = EXPLORATION_MAX
        self.steps = 0
    
    def select_action(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, num_actions - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                return self.dqn(state_tensor).argmax().item()

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return  # Skip if buffer is too small

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        q_values = self.dqn(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update Target Network
        if self.steps % TARGET_UPDATE == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
        
        # Decay exploration rate
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate * EXPLORATION_DECAY)
        self.steps += 1

# 🚀 Training Loop
num_actions = 6  # Pong has 6 discrete actions
env = PongEnv()
agent = DQNAgent((4, 84, 84), num_actions)

EPISODES = 5000
for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        agent.update()

    print(f"Episode {episode + 1}, Reward: {total_reward}, ε: {agent.exploration_rate:.3f}")

    # Save Model Every 100 Episodes
    if (episode + 1) % 100 == 0:
        torch.save(agent.dqn.state_dict(), "pong_dqn.pth")

print("Training complete! Model saved as `pong_dqn.pth`")
