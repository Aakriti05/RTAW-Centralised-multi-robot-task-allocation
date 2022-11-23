from astar import AStar
import sys
import math
import numpy as np


class astar_my(AStar):

    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, maze):
        #self.layout = np.loadtxt('/scratch/aakriti/complex_task_100rob_layout_newtask/warehouse/layouts/compact_2_64.cfg', skiprows=10)
        # self.lines = maze.strip().split('\n')
        self.layout = np.loadtxt(maze, skiprows=10)
        self.width = self.layout.shape[1]  #len(self.lines[0])
        self.height = self.layout.shape[0]  #len(self.lines)

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        return math.dist(n1, n2)

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        x, y = node
        return[(nx, ny) for nx, ny in [(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)] if 0 <= nx < self.width and 0 <= ny < self.height and self.layout[int(nx),int(ny)] == 0]
