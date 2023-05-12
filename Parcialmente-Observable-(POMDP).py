import numpy as np
from pomdpy.pomdp import Model
from pomdpy.solvers import POMCP
from pomdpy.util import console, config_parser

class POMDP(Model):
    def __init__(self):
        Model.__init__(self)

    def transition_function(self, state, action):
        next_state = np.random.choice([0, 1], p=[0.7, 0.3])
        return next_state

    def reward_function(self, state, action, next_state):
        if action == 1 and next_state == 1:
            return 10
        elif action == 1 and next_state == 0:
            return -10
        else:
            return 0

    def observation_function(self, state, action, next_state, observation):
        if next_state == 1:
            observation = 1
        else:
            observation = 0
        return observation

    def is_terminal(self, state):
        return False

    def reset(self):
        return 0

    def get_all_actions(self):
        return [0, 1]

    def get_all_observations(self):
        return [0, 1]

    def get_all_states(self):
        return [0, 1]

    def get_initial_belief_state(self):
        return [0.5, 0.5]

if __name__ == '__main__':
    pomdp = POMDP()

    solver_params = config_parser.parse_file("pomcp_example_params.conf")

    solver = POMCP(pomdp, **solver_params)
        
    for i in range(10):
        accion = solver.policy.get_best_action()
        obs = pomdp.get_observation(accion)
        console.info("Tomando la acción {} Observando {}".format(accion, obs))
        recompensa = pomdp.get_reward(accion, obs)
        console.info("Recompensa: {}".format(recompensa))
        solver.update(accion, obs, recompensa)
    print("Valor: ", solver.get_value(pomdp.get_initial_belief_state()))
