import numpy as np
from model import SelfLockingControlBinder, DQNN


GRID_SIZE = 128
MAX_ITERATION = 10000

DEFAULT_START_POSITION = [0, 0]
DEFAULT_GOAL_POSITION = [GRID_SIZE - 1, GRID_SIZE - 1]


def get_base_action(agent_position: list, goal_position: list) -> np.ndarray:
    
    action = np.zeros(4, dtype=int)
    
    if agent_position[0] < goal_position[0]:
        action[1] = 1
    elif agent_position[0] > goal_position[0]:
        action[0] = 1

    if agent_position[1] < goal_position[1]:
        action[3] = 1
    elif agent_position[1] > goal_position[1]:
        action[2] = 1
    
    return action


def get_action(state: np.ndarray, agent_position: list, goal_position: list) -> np.ndarray:
    
    action = get_base_action(agent_position, goal_position)
    
    # TODO
    # Modity base action according to past states and current state
    
    return action


def move_agent(agent_position: list, action: np.ndarray) -> list:

    up, down, left, right = action

    agent_position[0] += down - up
    agent_position[1] += right - left

    agent_position[0] = max(0, min(GRID_SIZE - 1, agent_position[0]))
    agent_position[1] = max(0, min(GRID_SIZE - 1, agent_position[1]))

    return agent_position


def get_initial_state(goal_position: list) -> np.ndarray:

    grid = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8) * 255
    grid[goal_position[0], goal_position[1]] = [0, 255, 0]

    return grid


def simulate(start_position: list=DEFAULT_START_POSITION, goal_position: list=DEFAULT_GOAL_POSITION):
    
    model = DQNN()
    wrapper = SelfLockingControlBinder(model)

    initial_state = get_initial_state(goal_position)
    agent_position = start_position

    for _ in range(MAX_ITERATION):

        state = initial_state.copy()
        state[agent_position[0], agent_position[1]] = [255, 0, 0]

        action = get_action(state, agent_position, goal_position)
        action = wrapper.rebind(state, action)
        agent_position = move_agent(agent_position, action)

        print(f"Agent Position: {agent_position}")

        if agent_position == goal_position:
            break

    print("Goal reached!")