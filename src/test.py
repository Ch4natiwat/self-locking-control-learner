from simulation import simulate
import random as rd


start_position = [rd.randint(0, 127), rd.randint(0, 127)]
goal_position = [rd.randint(0, 127), rd.randint(0, 127)]

simulate(start_position, goal_position)