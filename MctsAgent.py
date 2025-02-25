import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import cv2
import math
import ale_py
import copy


# --- Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99

# --- Preprocessing Function ---
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (84, 84))              # Resize
    return frame / 255.0                             # Normalize

class PongEnv:
    """
    Wraps 'ALE/Pong-v5' so that we keep a state_buffer of the last 4 frames
    and implement a .clone() and .restore() method that uses the underlying ALE
    environment's cloneSystemState/restoreSystemState.
    """
    def __init__(self, env_id="ALE/Pong-v5", frameskip=4, render_mode=None):
        self.env = gym.make(env_id, frameskip=frameskip, render_mode=render_mode)
        self.frameskip = frameskip
        self.state_buffer = collections.deque(maxlen=4)
        self.last_lives = None

    def reset(self):
        obs, _ = self.env.reset()
        processed = preprocess(obs)
        for _ in range(4):
            self.state_buffer.append(processed)
        return np.array(self.state_buffer, dtype=np.float32)

    def step(self, action):
        next_obs, reward, done, trunc, info = self.env.step(action)
        processed = preprocess(next_obs)
        self.state_buffer.append(processed)
        return np.array(self.state_buffer, dtype=np.float32), reward, (done or trunc)

    def clone(self):
        # 1. Unwrap the current env so we can grab the ALE state
        base_env = self.env.unwrapped
        ale_state = base_env.ale.cloneSystemState()

        # 2. Make a fresh PongEnv
        cloned_env = PongEnv(frameskip=self.frameskip)

        # 3. Call reset once so that the environment is "started"
        cloned_env.env.reset()

        # 4. Now restore the exact ALE state into the unwrapped env
        base_cloned = cloned_env.env.unwrapped
        base_cloned.ale.restoreSystemState(ale_state)

        # 5. Also copy your 4-frame buffer
        cloned_env.state_buffer = copy.deepcopy(self.state_buffer)

        return cloned_env

    def render(self):
        """
        Just call the underlying environment's render method.
        Note: Remember to create this environment with `render_mode='human'`
        if you want a visible game window.
        """
        return self.env.render()
    
    def close(self):
        self.env.close()


class MCTSNode:
    def __init__(self, state, parent=None, action=None, done=False):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.done = done  # track if terminal

    def uct_selection(self, c=1.4):
        best_score = float('-inf')
        best_children = []
        for child in self.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = c * math.sqrt(math.log(self.visits + 1) / child.visits)
                score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_children = [child]
            elif abs(score - best_score) < 1e-12:
                best_children.append(child)
        return random.choice(best_children)

    def expand(self, env):
        """
        Expand this node by adding a child node for each possible action
        from the *current* (cloned) environment state, rather than resetting.
        """
        if self.done:
            return  # No need to expand a terminal node

        num_actions = env.env.action_space.n
        for action in range(num_actions):
            # Don’t expand if we already made that child:
            if not any(child.action == action for child in self.children):
                # 1. Clone the environment from *this* node's game state
                temp_env = env.clone()

                # 2. Step with the given action
                next_state, reward, done = temp_env.step(action)

                # 3. Create the child node
                new_child = MCTSNode(
                    state=next_state,
                    parent=self,
                    action=action,
                    done=done
                )
                self.children.append(new_child)

    def backpropagate(self, reward):
        node = self
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

# --- MCTS Agent ---
class MCTSAgent:
    def __init__(self, env, dqn, simulations=10, c=1.4):
        self.env = env
        self.dqn = dqn
        self.simulations = simulations
        self.c = c

    def select_action(self, state):
        root = MCTSNode(state)
        for _ in range(self.simulations):
            node = root
            while node.children:
                node = node.uct_selection(self.c)
            node.expand(self.env)
            if node.children:
                chosen_child = random.choice(node.children)
                state_tensor = torch.tensor(chosen_child.state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.dqn(state_tensor)
                    action = q_values.argmax().item()
                chosen_child.backpropagate(q_values.max().item())
        if root.children:
            best_child = max(root.children, key=lambda child: child.visits)
            return best_child.action
        return random.randint(0, self.env.env.action_space.n - 1)  # Fallback

# --- Dummy DQN for Demonstration ---
class DummyDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DummyDQN, self).__init__()
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

# --- Main Function to Watch the Agent Play ---
if __name__ == "__main__":
    num_actions = 6
    dqn = DummyDQN((4, 84, 84), num_actions).to(DEVICE)
    
    env = PongEnv("ALE/Pong-v5", render_mode="human")  # Enable rendering
    state = env.reset()
    
    mcts_agent = MCTSAgent(env, dqn, simulations=10, c=1.4)
    
    done = False
    while not done:
        action = mcts_agent.select_action(state)
        state, reward, done = env.step(action)
        env.render()  # Render game frame

    env.close()  # Close the environment
