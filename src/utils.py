
import pickle

def save_model(agent, filename):
    with open(filename, 'wb') as f:
        pickle.dump(agent.q_table, f)
        print(f"Model saved to {filename}")
