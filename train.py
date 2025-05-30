import torch
from main import env_run, Mymodel_action, OffPolicyPPO

def train_model():
    # Initialize the action model
    action_model = Mymodel_action()

    # Initialize the environment
    env = env_run(action_model=action_model, memory_size=10000)  # Added memory_size parameter

    # Initialize the PPO algorithm
    ppo = OffPolicyPPO(
        lr=3e-4,          # Learning rate
        gamma=0.99,       # Discount factor
        clip_ratio=0.2,   # Clipping ratio for PPO
        entropy_coef=0.01 # Entropy coefficient
    )

    # Training parameters
    num_episodes = 1000      # Number of episodes
    max_steps = 500          # Max steps per episode
    batch_size = 32          # Batch size for training
    buffer_size = 10000      # Replay buffer size

    # Train the model
    print("Starting training...")
    ppo.train(env, num_episodes, max_steps, batch_size, buffer_size)

    # Save the trained model
    torch.save(action_model.state_dict(), "action_model.pth")
    print("Training complete. Model saved as 'action_model.pth'.")

if __name__ == "__main__":
    train_model()