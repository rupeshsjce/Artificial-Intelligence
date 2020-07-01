#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:33:30 2020

@author: Rupesh Kumar
"""

import sys
from collections import deque

#to use for priority queue
import heapq
import operator
import copy
import time

def directionVector_add(a, b):
    return tuple(map(operator.add, a, b))

#______________________________________________________________________________   
def printOCtilePath(problem, state, start, end):
    
    algo = problem.algo
    W = problem.W
    H = problem.H
    best = ()
    best_action = ""
    
    dx = abs(start[0] - end[0])
    dy = abs(start[1] - end[1])
    D1, D2 = 1, 1
    
    Do_initial = D1 * (dx + dy) + (D2 - 2 * D1) * min(dx, dy)
    if algo == "BFS":
        D1, D2 = 1,1
    else:    
        D1, D2 = 10, 14
       
    Do_min = D1 * (dx + dy) + (D2 - 2 * D1) * min(dx, dy)
    #print("Do_initial : ", Do_initial)
    
    for i in range(Do_initial):        
        for action in allowed_nj_actions:
            next_node_xy = directionVector_add((start[0], start[1]), 
                                               directions[action])
            dx = abs(next_node_xy[0] - end[0])
            dy = abs(next_node_xy[1] - end[1])
            """
              0 < = next_node_xy[0] <= W-1 and 0 <= next_node_xy[1] <= H-1
            """
            if not (next_node_xy[0] >=0 and next_node_xy[0] <= W-1 and next_node_xy[1] >= 0 and next_node_xy[1] <= H-1):
                continue
            
            #D1, D2 = 10, 14
            Do = D1 * (dx + dy) + (D2 - 2 * D1) * min(dx, dy)
            if (Do < Do_min):
                Do_min = Do
                best = next_node_xy
                best_action = action

        start = best        
        print(state, start[0], start[1], cost[algo][best_action])
        f.write('\n' + str(state) + " " + str(start[0]) + " " + 
                    str(start[1]) + " " + str(cost[algo][best_action]))        
        
    return None        
#______________________________________________________________________________
def printPathBwJauntNodes(problem, path_node):
    print(path_node[0].state, path_node[0].x, path_node[0].y, path_node[0].state_cost)
    f.write('\n' + str(path_node[0].state) + " " + str(path_node[0].x) + " " +
            str(path_node[0].y) + " " + str(path_node[0].state_cost))
    # use curr and next node
    for i in range(len(path_node) - 1):
        curr = path_node[i]
        next = path_node[i+1]
        if curr.state == next.state:
            printOCtilePath(problem, curr.state, (curr.x,curr.y), (next.x, next.y))
        else:
            print(next.state, next.x, next.y, next.state_cost)
            f.write('\n' + str(next.state) + " " + str(next.x) + " " + 
                    str(next.y) + " " + str(next.state_cost))
            
    return None        

#______________________________________________________________________________
def findNode(key_node, frontier):
    for node in frontier:
        if node.state == key_node.state and node.x == key_node.x and node.y == key_node.y and node.jaunt_end == key_node.jaunt_end:
            return node
    return None              
#______________________________________________________________________________        
def printPath(problem, node):
    path_node = node.path()
    print(node.path_cost)
    f.write(str(node.path_cost) + '\n')
    total_step = 1 + path_node[-1].depth
    print(total_step)
    f.write(str(total_step))
    printPathBwJauntNodes(problem, path_node)   
    return None 

#______________________________________________________________________________
       
class Node:

    def __init__(self, state, x=0, y=0, parent=None, action=None, depth=0, 
                 path_cost=0,state_cost=0, total_cost=0,jaunt_end=None):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state # YEAR
        self.jaunt_end = jaunt_end
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.state_cost = state_cost
        self.total_cost = total_cost
        self.x = x
        self.y = y
        self.depth = depth
        
    def InFrontierExplored(self, frontier_explored):
        for node in frontier_explored:
            if (node.state == self.state and node.x == self.x and node.y == self.y and node.jaunt_end== self.jaunt_end):
                return True
            
        return False     
 
    def Union(self, lst1, lst2): 
        final_list = list(set(lst1) | set(lst2)) 
        return final_list 
    
    """
      jauntNodes : is a list[] with all the jauntNodes which are reachable.
      return all the nodes which are having same state as self.state and other channel end-points if exist
    """      
    def findAllowedActions(self, jauntNodes):
        """
          find all the (self.state, self.x, self.y, self.jaunt_end) in jauntNodes 
          and return all nodes from same or other world
        """
        jauntingNode = [] # list with all the destination node from self node
        jauntingNode1 = [] 
        jauntingNode2 = []
        for node in jauntNodes:
            if (node.state == self.state):
                jauntingNode1.append(node)
            if (node.jaunt_end == self.state and node.x == self.x and node.y == self.y):
                jauntingNode1.append(node)
          
        jauntingNode = self.Union(jauntingNode1, jauntingNode2)   
        return jauntingNode
        
        
    def expand(self, problem):
        """List the jaunt nodes reachable from this node."""
        return [self.neighbor_node(problem, action)
                for action in problem.actions(self)]  # action is one of the jauntNodes

    def neighbor_node(self, problem, action):
        """neighbor_node are all the jaunt Nodes in the same world or different world."""
        next_node = problem.result(self, action)
        return next_node

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __repr__(self):
        return "<Node {0}, {1}, {2}, {3}, {4}, {5}, {6}>".format(self.state, self.x, self.y, self.jaunt_end, self.state_cost, self.path_cost,self.total_cost)
    
    def __lt__(self, node):
        return self.total_cost < node.total_cost    

#______________________________________________________________________________       
def uniform_cost_search(problem):
                
    node = problem.initial_node    
    if problem.goal_test(node):
        printPath(problem, node)
        return None
    
    frontier = []
    heapq.heapify(frontier) 
    heapq.heappush(frontier, node)
        
    explored = []
    while frontier:
        node = heapq.heappop(frontier)
        if problem.goal_test(node):
           printPath(problem, node)
           return None
       
        explored.append(node)
        for next_node in node.expand(problem):
            if not next_node.InFrontierExplored(explored) and not next_node.InFrontierExplored(frontier):
                heapq.heappush(frontier, next_node)
            elif next_node.InFrontierExplored(frontier):
                nodeF = findNode(next_node, frontier)
                if next_node.total_cost < nodeF.total_cost:
                    frontier.remove(nodeF) # 
                    heapq.heapify(frontier) 
                    heapq.heappush(frontier, next_node)
    print("FAIL") 
    f.write("FAIL")               
    return None

#______________________________________________________________________________
"""uniform_cost_search with f(n) = g(n) + h(n) is A* """
def ASTAR_search(problem):
    uniform_cost_search(problem)

#______________________________________________________________________________
"""uniform_cost_search with sorting queue based on depth(this code is using path_cost/total_cost as depth) is BFS. """
def BFS_search(problem):
    uniform_cost_search(problem)

#______________________________________________________________________________
directions = {'NW': (-1, 1), 'NE': (1, 1), 'SE': (1, -1), 'SW': (-1, -1),
              'W': (-1, 0), 'N': (0, 1), 'E': (1, 0), 'S': (0, -1)}
               

allowed_nj_actions = ["NE","NW","SE","SW","N","S","E","W"]

cost = dict(BFS = dict(N=1,S=1,E=1,W=1,NE=1,NW=1,SE=1,SW=1),
            UCS = dict(N=10,S=10,E=10,W=10,NE=14,NW=14,SE=14,SW=14),
            ASTAR  = dict(N=10,S=10,E=10,W=10,NE=14,NW=14,SE=14,SW=14))


#______________________________________________________________________________
class BackToTheFutureProblem():
    """Problem of travelling from initial to final destination in a time travelling machine """
    
    """
       algo       : BFS, UCS, ASTAR
       grid       : grid(H,W)
       initial    : START [2020,3,4]
       target     : GOAL  [2080,6,7]
       noc        : number of bi-directional channels
       channels   : All the channels in the form os nested lists. [[2020,10,11,2040],[2040,34,12,2080]]
       defined_actions : direction8 for all algo. Additionally "Jaunt" is taken care separately
    
    """

    def __init__(self, algo, grid, initial, target, noc, channels, defined_actions=directions):
        """The grid is a 2 dimensional array/list whose state is specified by tuple of indices"""
        """ grid is grid(W, H) W = Width and H = Height"""
        self.algo = algo[0]
        self.grid = grid
        self.initial_node = Node(initial[0], initial[1], initial[2], jaunt_end=initial[0])
        self.target_node =  Node(target[0], target[1], target[2], jaunt_end=target[0])
        self.noc = noc
        self.channels = channels
        self.defined_actions = defined_actions
        self.W = grid[0] # Width or 0 <= X <= W-1
        self.H = grid[1] # Height or 0 <= Y <= H-1
        self.jauntNodes = []
        self.print = True

    def FormJauntNodes(self):
        jauntNodes = [] # all the jaunt end points with distinct(year,x,y)
        # Make START node and GOAL node as JauntNodes also with jaunt_end = self
        jauntNodes.append(Node(self.initial_node.state, self.initial_node.x, self.initial_node.y, jaunt_end = self.initial_node.state))
        jauntNodes.append(Node(self.target_node.state, self.target_node.x, self.target_node.y, jaunt_end = self.target_node.state))    
        for i in range(self.noc):
            jaunt = self.channels[i]
            # create two nodes from jaunt as these are jaunt end-point nodes
            jauntNodes.append(Node(jaunt[0], jaunt[1], jaunt[2], jaunt_end=jaunt[3]))
            jauntNodes.append(Node(jaunt[3], jaunt[1], jaunt[2], jaunt_end=jaunt[0]))
            
        self.jauntNodes = jauntNodes
        return None

    def actions(self, node):
        """Returns the list of jaunt nodes which are allowed to be taken from the given node(state, x, y)
        """
        allowed_actions = []
        # allowed actions could be none or one or more jauntNodes
        # a node location can have None, one or more Jaunt end-point to other world.
        allowed_actions = node.findAllowedActions(self.jauntNodes) # if returned None then no action
        return allowed_actions

    def result(self, node, action):
        
        dx = abs(node.x - action.x)
        dy = abs(node.y - action.y)
        Dw = (abs(node.state - action.state))
        
        next_node = copy.deepcopy(action) # action where we want to go
        next_node.parent = node
        next_node.action = action
        
        D1, D2 = 1,1
        depth = D1 * (dx + dy) + (D2 - 2 * D1) * min(dx, dy)
        
        if self.algo == "BFS":
            D1, D2 = 1, 1
        else: 
            D1, D2 = 10,14
            
        Do = D1 * (dx + dy) + (D2 - 2 * D1) * min(dx, dy)

        
        # state cost is octile distance for same world
        if node.state == action.state: 
            next_node.depth = node.depth + depth
            next_node.state_cost = Do 
        else:
            next_node.depth = node.depth + 1
            if self.algo == "BFS":
                 next_node.state_cost = 1
            else: 
                 next_node.state_cost = Dw
              
        next_node.path_cost = node.path_cost + next_node.state_cost

        """
              for A* search, let add h(n) to path_cost. Adding few lines for that. 
              Other than this, no need to change __lt__ (Node) and UCS algo.
              Final cost will be addition of individual node state cost.
        """
        next_node.total_cost = next_node.path_cost + self.h(next_node)
        # returned node will be in same world or different world
        return next_node    

    def goal_test(self, node):
        """Return True if the node is a goal. 
           It should match (state,x,y)
        """
        return (node.state == self.target_node.state and node.x == self.target_node.x and node.y == self.target_node.y)  
  
    def h(self, node):
        """ Return the heuristic value for a given state. Default heuristic function used is 
        h(n) = {
           Octile distance = D1 * (dx + dy) + (D2 - 2 * D1) * min(dx, dy) if node and Goal are in same world
           Octile distance + abs(node.state - goal.state) ; if node and Goal are in different world
        }
        h(n) = 0 for BFS and UCS algo.
        """
        if self.algo == "ASTAR":
           # D1 = N,S,E,W distance = 10; D2 = NE,NW,SE,SE = 14 
           D1, D2 = 10, 14
           
           dx = abs(node.x - self.target_node.x)
           dy = abs(node.y - self.target_node.y)
           Do = D1 * (dx + dy) + (D2 - 2 * D1) * min(dx, dy)
           Dw = (abs(node.state - self.target_node.state)) 
           
           if node.state != self.target_node.state:
               return Do + Dw
           else:
               return Do
              
        else:
            return 0

#______________________________________________________________________________

import json
 
params = []
channels = []
count = 0
addL = 5

f = open("input.txt", "r").read().split('\n')
for line in f:
     
  l = line.replace(' ', ',')
  if count == 0:     
      if l == "A*":
          l = "ASTAR"
      l = "[\"{0}\"]".format(l)    
  elif count == 4: # number of channels
      l = "{0}".format(l) 
      addL = count + int(line)
  else:
      l = "[{0}]".format(l) 
 
  l = json.loads(l)
  
  #index 5 ownwards
  if 5 <= count <= addL:
      channels.append(l)
      
  params.append(l)
  count = count + 1 
  
#______________________________________________________________________________
if (count < 5):
    raise ValueError("Invalid Input")
 

searchAlgorithms = {'BFS': BFS_search, 'UCS': uniform_cost_search, 
                    'ASTAR': ASTAR_search}


f = open("output.txt", "w")

start_time = time.time()
problem = BackToTheFutureProblem(params[0], params[1], params[2], 
                                 params[3], params[4], channels)
#print(params[0])
problem.FormJauntNodes()
searchAlgorithms[problem.algo](problem)

print("Execution time : %s seconds\n" % (time.time() - start_time))

f.close()
#______________________________________________________________________________
