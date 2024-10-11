from maze_env import MazeEnv
from q_learning_agent import QLearningAgent
import matplotlib.pyplot as plt

def train_agent(episodes=1000):
    env = MazeEnv()
    agent = QLearningAgent(state_size=(10, 10), action_size=4)
    rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        if (episode + 1) % 100 == 0:
            agent.save('models/trained_agent.pkl')

    env.render()

    # Plot total rewards over episodes
    plt.plot(rewards)
    plt.title('Total Rewards Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.show()

def test_agent():
    env = MazeEnv()
    agent = QLearningAgent(state_size=(10, 10), action_size=4)
    agent.load('models/trained_agent.pkl')
    
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = np.argmax(agent.q_table[state[0], state[1], :])  # Choose the best action
        state, reward, done = env.step(action)
        total_reward += reward
        env.render()  # Optionally render the environment after each step

    print(f"Test Total Reward: {total_reward}")

if __name__ == '__main__':
    train_agent()
    test_agent()