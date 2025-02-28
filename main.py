import gymnasium as gym
from gymnasium.wrappers import RecordVideo  # Para gravar vídeos (opcional)

# Cria o ambiente Pusher-v4 com renderização em janela humana
env = gym.make(
    "Pusher-v4", render_mode="human"
)  # Modo de renderização para visualização ao vivo

# Se quiser gravar vídeos:
# env = RecordVideo(gym.make('Pusher-v4'), video_folder="videos", name_prefix="pusher_run")

observation, info = env.reset()

num_episodes = 1_000_000

for _ in range(num_episodes):
    # Especifica uma ação aleatória
    action = env.action_space.sample()

    # Executa a ação no ambiente
    observation, reward, terminated, truncated, info = env.step(action)

    # Verifica se o episódio terminou
    if terminated or truncated:
        print(f"Episódio terminado com recompensa: {reward}")
        observation, info = env.reset()

# Fecha o ambiente
env.close()
