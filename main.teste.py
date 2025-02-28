import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy

# Configurações principais
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 1000
BATCH_SIZE = 64
BUFFER_CAPACITY = 10000


# Classe do Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices]
        )
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.FloatTensor(np.array(actions)).to(DEVICE),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(DEVICE),
        )


# Configuração do ACT para o Pusher-v4
def get_act_config(env):
    return ACTConfig(
        input_shapes={"observation.state": [env.observation_space.shape[0]]},
        output_shapes={"action": [env.action_space.shape[0]]},
        use_vae=False,
        n_obs_steps=1,
        chunk_size=10,
        n_action_steps=10,
        dim_model=256,
        n_heads=4,
        n_encoder_layers=3,
        optimizer_lr=1e-4,
        normalization_mapping={
            "VISUAL": "none",
            "STATE": "mean_std",
            "ACTION": "mean_std",
        },
    )


# Inicialização do ambiente e política
env = gym.make("Pusher-v4", render_mode="human")
config = get_act_config(env)
# policy = ACTPolicy(config).to(DEVICE)
policy = ACTPolicy.from_pretrained("lerobot/pi0", config=config).to(DEVICE)
optimizer = torch.optim.Adam(policy.parameters(), lr=config.optimizer_lr)
buffer = ReplayBuffer(BUFFER_CAPACITY)

# Loop de treinamento
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0
    policy.reset()

    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            action_tensor = policy.select_action({"observation.state": state_tensor})
            action = action_tensor.cpu().numpy()[0]

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # Atualização do modelo
        if len(buffer.buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            # Forward pass
            loss, loss_dict = policy(
                {
                    "observation.state": states,
                    "action": actions,
                    "action_is_pad": torch.zeros_like(dones, dtype=torch.bool),
                }
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        if done:
            print(
                f"Episódio {episode+1}, Recompensa: {episode_reward:.1f}, Loss: {loss.item():.4f}"
            )
            break

env.close()
