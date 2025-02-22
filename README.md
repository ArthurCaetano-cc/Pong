# Reinforcement Learning Atari Agent: Pong with DQN and MCTS

This project implements a Reinforcement Learning (RL) agent capable of playing the classic Atari game "Pong" using a combination of Deep Q-Network (DQN), Convolutional Neural Networks (CNNs), and Monte-Carlo Tree Search (MCTS). The goal of this project is to explore how these approaches, individually and collectively, enhance an agent's ability to learn complex decision-making tasks directly from pixel data.

## Project Goals
- Develop and train RL agents using Deep Q-Networks (DQN).
- Apply Convolutional Neural Networks (CNN) to process raw pixel input from the Atari environment.
- Integrate Monte-Carlo Tree Search (MCTS) to improve decision-making and planning.
- Analyze the performance of combined RL methods against single-method baselines.

## Components

### Deep Q-Network (DQN)
DQN is employed to learn optimal policies directly from high-dimensional sensory inputs (frames of the Atari game). The CNN architecture processes raw image inputs and outputs action-values.

### Convolutional Neural Networks (CNN)
CNNs are utilized to automatically extract meaningful features from raw pixel data, which allows the agent to understand visual information and significantly enhances the generalization and efficiency of training.

### Monte-Carlo Tree Search (MCTS)
MCTS is implemented to provide an efficient exploration-exploitation strategy, guiding the agent by simulating possible future actions to select the best immediate moves.

## Dependencies
- Python
- OpenAI Gym
- ALE (Arcade Learning Environment)
- NumPy
- OpenCV (cv2)
- PyGame (for visualization)

## Setup and Execution
1. Install required libraries:
```bash
pip install gym[atari] ale-py gym[accept-rom-license] opencv-python pygame
```

2. Run the agent:
```bash
python your_script.py
```

## Evaluation and Results
Detailed analyses of performance metrics and comparisons across methods are included in the technical report accompanying this repository.

## Contribution
Pull requests and discussions are welcomed to enhance and extend the functionalities provided here.

## Authors
- Your Name
- Team Members

## Acknowledgements
This project is inspired by seminal works in Deep Reinforcement Learning by Mnih et al. and the integration of planning techniques like MCTS by Guo et al. as benchmarks for Atari games.

## References
- [Playing Atari with Deep Reinforcement Learning (DQN)](https://arxiv.org/abs/1312.5602)
- [Deep Learning for Real-Time Atari Game Play Using Offline MCTS](https://paperswithcode.com/paper/deep-learning-for-real-time-atari-game-play)

Feel free to reach out if you have any questions or suggestions!


