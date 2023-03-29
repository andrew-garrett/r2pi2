# CIS 521: R2D2 - Homework 2
from typing import List, Tuple, Set, Optional
from collections import deque
from queue import PriorityQueue
from math import sqrt
from itertools import permutations
import time

import numpy as np
from spherov2 import scanner
from spherov2.sphero_edu import SpheroEduAPI

student_name = 'Andrew Todd Garrett'

Vertex = Tuple[int, int]
Edge = Tuple[Vertex, Vertex]


# Part 1: Compare Different Searching Algorithms
class Graph:
    """A directed Graph representation"""

    def __init__(self, vertices: Set[Vertex], edges: Set[Edge]):
        self.vertices = vertices
        self.N = len(vertices)
        self.edges = {} # adjacency dict
        self.M = len(edges)
        for edge in edges:
            if edge[0] not in self.edges.keys():
                self.edges[edge[0]] = set([edge[1]])
            else:
                self.edges[edge[0]].add(edge[1])
        return;

    def neighbors(self, u: Vertex) -> Set[Vertex]:
        """Return the neighbors of the given vertex u as a set"""
        if u not in self.edges.keys():
            return set();
        return self.edges[u];

    def backtrack(self, parents, start, goal):
        current = goal
        path = []
        path.append(current)
        while current is not start:
            prev = parents[current]
            path.append(prev)
            current = prev
        path.reverse()
        return path;

    def bfs(self, start: Vertex, goal: Vertex) -> Tuple[Optional[List[Vertex]], Set[Vertex]]:
        """Use BFS algorithm to find the path from start to goal in the given graph.

        :return: a tuple (shortest_path, node_visited),
                 where shortest_path is a list of vertices that represents the path from start to goal, and None if
                 such a path does not exist; node_visited is a set of vertices that are visited during the search."""
        queue = deque()
        visited = set()
        parents = {}
        visited.add(start)
        queue.append(start)
        parents[start] = None
        while queue:
            current = queue.popleft()
            visited.add(current)
            if current == goal:
                path = self.backtrack(parents, start, goal)
                return (path, visited);
            for neighbor in self.neighbors(current):
                if neighbor not in visited:
                    queue.append(neighbor)
                    parents[neighbor] = current
        return ([], visited);

    def dfs(self, start: Vertex, goal: Vertex) -> Tuple[Optional[List[Vertex]], Set[Vertex]]:
        """Use BFS algorithm to find the path from start to goal in the given graph.

        :return: a tuple (shortest_path, node_visited),
                 where shortest_path is a list of vertices that represents the path from start to goal, and None if
                 such a path does not exist; node_visited is a set of vertices that are visited during the search."""
        stack = deque()
        visited = set()
        parents = {}
        visited.add(start)
        stack.append(start)
        parents[start] = None
        while stack:
            current = stack.pop()
            visited.add(current)
            if current == goal:
                path = self.backtrack(parents, start, goal)
                return (path, visited);
            for neighbor in self.neighbors(current):
                if neighbor not in visited:
                    stack.append(neighbor)
                    parents[neighbor] = current
        return (None, visited);
    
    def heuristic(self, node, goal):
        # Manhattan Distance Heuristic
        h = abs(node[0] - goal[0])
        h += abs(node[1] - goal[0])
        #h *= 10.0
        return int(h);

    def a_star(self, start: Vertex, goal: Vertex) -> Tuple[Optional[List[Vertex]], Set[Vertex]]:
        """Use A* algorithm to find the path from start to goal in the given graph.

        :return: a tuple (shortest_path, node_visited),
                 where shortest_path is a list of vertices that represents the path from start to goal, and None if
                 such a path does not exist; node_visited is a set of vertices that are visited during the search."""
        frontier = PriorityQueue()
        visited = set()
        parents = {}
        g_scores = {}
        for v in self.vertices:
            g_scores[v] = 1e10
        g_scores[start] = 0
        f = g_scores[start] + self.heuristic(start, goal)
        frontier.put((f, start))
        visited.add(start)
        parents[start] = None

        while not frontier.empty():
            _, current = frontier.get()
            visited.add(current)
            if current == goal:
                path = self.backtrack(parents, start, goal)
                return (path, visited);
            for neighbor in self.neighbors(current):
                g = g_scores[current] + 1
                if g < g_scores[neighbor]:
                    parents[neighbor] = current
                    g_scores[neighbor] = g
                    f = g_scores[neighbor] + self.heuristic(neighbor, goal)
                    if neighbor not in visited:
                        frontier.put((f, neighbor))
        return (None, visited);

    def tsp(self, start: Vertex, goals: Set[Vertex]) -> Tuple[Optional[List[Vertex]], Optional[List[Vertex]]]:
        """Use A* algorithm to find the path that begins at start and passes through all the goals in the given graph,
        in an order such that the path is the shortest.

        :return: a tuple (optimal_order, shortest_path),
                 where shortest_path is a list of vertices that represents the path from start that goes through all the
                 goals such that the path is the shortest; optimal_order is an ordering of goals that you visited in
                 order that results in the above shortest_path. Return (None, None) if no such path exists."""
        
        paths = {}
        for perm in permutations(goals, len(goals)):
            path, _ = self.a_star(start, perm[0]) # find A* path from start to first goal in permutation
            path = path[:-1]
            prev = perm[0]
            if path is not None: # if there exists a path, continue searching
                counter = 1 # count the number of goals successfully visited
                for node in perm[1:]:
                    temp, _ = self.a_star(prev, node)
                    if temp is not None:
                        if counter == len(goals) - 1: # last node in path
                            path.extend(temp)
                        else:
                            path.extend(temp[:-1])
                            prev = node
                        counter += 1
                    else:
                        break;
                if counter == len(goals):
                    paths[perm] = path
        if len(paths.keys()) > 0:
            min_order = min(paths.keys(), key=(lambda k: len(paths[k])))
            min_path = paths[min_order]
            return (min_order, min_path);
        return (None, None);


# Part 2: Let your R2-D2 rolling in Augment Reality (AR) Environment
def get_transformation(k: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calculate the transformation matrix using the given equation P = K x (R | T)"""
    r_t = np.hstack((r, t))
    p = np.matmul(k, r_t)
    return p;


def convert_3d_to_2d(p: np.ndarray, points_3d: List[Tuple[float, float, float]]) -> List[Tuple[int, int]]:
    """Convert a list of 3D real world points to 2D image points in pixels given the transformation matrix,
       preserving the order of the points."""
    points_2d = []
    for point_3d in points_3d:
        x, y, z = point_3d
        modded = np.array([x, y, z, 1.0])
        point = np.matmul(p, modded)
        new_z = point[2]
        point = (point / new_z).astype(int)
        points_2d.append(tuple(point[:2]))
    return points_2d;


def convert_2d_to_relative(point_2d: Tuple[int, int], maze_in_2d: List[List[Tuple[int, int]]]) -> Optional[Vertex]:
    """Convert a 2D image point to maze coordinates using the given maze coordinates in 2D image.
       Return None if the 2D point isn't in the maze. Assume the coordinates are axis-aligned."""
    n = len(maze_in_2d)
    m = len(maze_in_2d[0])
    x_point, y_point = point_2d
    for row in range(n-1):
        for col in range(m-1):
            b_box = (maze_in_2d[row][col], maze_in_2d[row+1][col+1])
            if x_point >= b_box[0][0] and x_point < b_box[1][0]:
                if y_point >= b_box[0][1] and y_point < b_box[1][1]:
                    return (row, col);


def path_to_moves(path: List[Vertex]) -> List[Tuple[int, int]]:
    """Taking a list of vertices and returns a list of droid actions (heading, steps)"""

    heading_mapping = {
        (0, 1): 90,
        (0, -1): 270,
        (1, 0): 180,
        (-1, 0): 0
    }
    actions = []
    if path is not None or len(path) > 0:
        prev = path[0]
        for node in path[1:]:
            dir = (node[0] - prev[0], node[1] - prev[1])
            if dir != (0, 0):
                heading = heading_mapping[dir]
                if len(actions) == 0:
                    actions.append((heading, 1))
                else:
                    last_action = actions[-1]
                    if heading == last_action[0]:
                        actions[-1] = (heading, last_action[1] + 1)
                    else:
                        actions.append((heading, 1))
            prev = node
    return actions;


def droid_roll(path: List[Vertex]):
    """Make your droid roll with the given path. You should decide speed and time of rolling each move."""
    moves = path_to_moves(path)
    print(moves)
    with SpheroEduAPI(scanner.find_R2D2(timeout=30)) as droid:
        prev_move = None
        for move in moves:
            if prev_move is None:
                droid.roll(move[0], 72, move[1])
            else:
                droid.set_heading(move[0])
                time.sleep(0.5)
                droid.roll(move[0], 72, move[1])
            droid.stop_roll()
            time.sleep(0.5)
            prev_move = move
        return;


# testing graph
#vertices = set((i, j) for i in range(10) for j in range(10))
#edges = set()
#for vertex in vertices:
#    for move in ((1, 0), (0, 1), (-1, 0), (0, -1)):
#        next_vertex = vertex[0] + move[0], vertex[1] + move[1]
#        if next_vertex in vertices:
#            edges.add((vertex, next_vertex))
#for f, t in (((3, 7), (3, 8)), ((4, 7), (4, 8)), ((5, 7), (5, 8)), ((6, 7), (6, 8)), ((7, 7), (7, 8)),
#                ((7, 3), (8, 3)), ((7, 4), (8, 4)), ((7, 5), (8, 5)), ((7, 6), (8, 6)), ((7, 7), (8, 7))):
#    edges.discard((f, t))
#    edges.discard((t, f))

#test_graph = Graph(vertices, edges)
#start = (6,1)
#goal = (9,9)
#path, visited = test_graph.bfs(start, goal)
#print(path)
#print(visited)
#unvisited = vertices.difference(visited)
#print(unvisited)
#print()
#path, visited = test_graph.dfs(start, goal)
#print(path)
#print(visited)
#unvisited = vertices.difference(visited)
#print(unvisited)
#print()
#path, visited = test_graph.a_star(start, goal)
#print(path)
#print(visited)
#unvisited = vertices.difference(visited)
#print(unvisited)
#print()
#goals = set([goal, (1, 9), (9, 1)])
#optimal_order, path = test_graph.tsp(start, goals)
#print(optimal_order)
#print(path)
#print(visited)
#unvisited = vertices.difference(visited)
#print(unvisited)


#################### AR Test ####################

#K = np.array([[1388., 0., 507.], [0., 1398., 364.], [0., 0., 1.]])
#R = np.random.rand(3,3)
#T = np.random.rand(3, 1)
#P = get_transformation(K, R, T)
#print(P.shape)
#points_3d = [(10.0, 10.0, 4.0), 
#             (-10.0, 10.0, 4.0), 
#             (10.0, -10.0, 4.0), 
#             (-10.0, -10.0, 4.0)]
#points_2d = convert_3d_to_2d(P, points_3d)
#print(points_2d)

#path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
#movement = path_to_moves(path)
#print(movement)

#droid_roll(path)