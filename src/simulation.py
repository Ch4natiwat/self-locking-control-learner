GRID_SIZE = 128
MAX_ITERATION = 10000

DEFAULT_START_POSITION = [0, 0]
DEFAULT_GOAL_POSITION = [GRID_SIZE - 1, GRID_SIZE - 1]


def get_direction() -> tuple:
    
    return (0, 1, 0, 1)


def move_agent(agent_position: list, action: tuple) -> list:
    
    up, down, left, right = action

    agent_position[0] += down - up
    agent_position[1] += right - left

    agent_position[0] = max(0, min(GRID_SIZE - 1, agent_position[0]))
    agent_position[1] = max(0, min(GRID_SIZE - 1, agent_position[1]))

    return agent_position


def simulate(start_position: list=DEFAULT_START_POSITION, goal_position: list=DEFAULT_GOAL_POSITION):
    
    agent_position = start_position
    
    for _ in range(MAX_ITERATION):
        
        action = get_direction()
        agent_position = move_agent(agent_position, action)

        print(f"Agent Position: {agent_position}")
        
        if agent_position == goal_position:
            break

    print("Goal reached!")