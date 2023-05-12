import gym
import numpy as np

# Crear el entorno FrozenLake-v0
env = gym.make('FrozenLake-v0')

# Inicializar la tabla de valores de utilidad y políticas aleatorias
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions)) # Inicializa la tabla de valores de utilidad en 0 para cada estado-acción
policy = np.ones(num_states, dtype=int) * num_actions # Inicializa una política aleatoria

# Establecer los parámetros del algoritmo de iteración de políticas
discount_factor = 0.99
num_iterations = 10000

# Algoritmo de iteración de políticas
for i in range(num_iterations):
    # Evaluar la política actual
    V = np.zeros(num_states) # Inicializa los valores de utilidad en 0 para cada estado
    for s in range(num_states):
        for a in range(num_actions):
            # Accede a las transiciones posibles desde el estado-acción actual
            next_states, rewards, done, _ = env.env.P[s][a]
            for j in range(len(next_states)):
                next_state = next_states[j]
                reward = rewards[j]
                # Acumula la utilidad esperada para el estado actual y acción actual de acuerdo a la política actual
                V[s] += policy[s] == a * (reward + discount_factor * Q[next_state, policy[next_state]])
    # Mejorar la política actual
    policy_stable = True
    for s in range(num_states):
        old_action = policy[s]
        q_values = []
        for a in range(num_actions):
            # Accede a las transiciones posibles desde el estado-acción actual
            next_states, rewards, done, _ = env.env.P[s][a]
            q_value = 0
            for j in range(len(next_states)):
                next_state = next_states[j]
                reward = rewards[j]
                # Acumula la utilidad esperada para cada posible acción en el estado actual
                q_value += (reward + discount_factor * Q[next_state, policy[next_state]])
            q_values.append(q_value)
        # Selecciona la acción con la utilidad esperada más alta
        new_action = np.argmax(q_values)
        if old_action != new_action:
            policy_stable = False
        policy[s] = new_action
    # Salir del bucle si se ha encontrado una política estable
    if policy_stable:
        break

# Probar la política aprendida
rewards = []
for i in range(100):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy[obs]
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    rewards.append(total_reward)
print("Recompensas promedio en 100 episodios:", np.mean(rewards))
