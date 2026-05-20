import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================
# ENVIRONMENT
# =========================

states = 9
actions = 4

# Actions
# 0 = up
# 1 = down
# 2 = left
# 3 = right

# Q TABLE
Q = np.zeros((states, actions))

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 100

# Rewards
goal_state = 8

# =========================
# NEXT STATE FUNCTION
# =========================

def get_next_state(state, action):

    row = state // 3
    col = state % 3

    if action == 0 and row > 0:
        row -= 1

    elif action == 1 and row < 2:
        row += 1

    elif action == 2 and col > 0:
        col -= 1

    elif action == 3 and col < 2:
        col += 1

    return row * 3 + col

# =========================
# TRAINING
# =========================

def train_agent():

    rewards_per_episode = []

    for episode in range(episodes):

        state = 0
        total_reward = 0
        done = False

        while not done:

            # Exploration vs exploitation
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(actions)
            else:
                action = np.argmax(Q[state])

            next_state = get_next_state(state, action)

            # Rewards
            if next_state == goal_state:
                reward = 10
                done = True
            else:
                reward = -1

            # Q-learning formula
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

    generate_graph(rewards_per_episode)

    return rewards_per_episode

# =========================
# GRAPH
# =========================

def generate_graph(rewards):

    plt.figure(figsize=(8,5))

    plt.plot(rewards)

    plt.title("Reward Progression")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")

    plt.grid()

    plt.savefig("static/rewards_graph.png")

    plt.close()

# =========================
# RESULTS
# =========================

def get_results():

    rewards = train_agent()

    policy = np.argmax(Q, axis=1)

    return {
        "qtable": Q.tolist(),
        "policy": policy.tolist(),
        "rewards": rewards[-10:],
        "total_reward": sum(rewards)
    }