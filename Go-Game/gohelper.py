import enum
from collections import namedtuple
import operator
#from goboardhash import *

def directionVector_add(a, b):
    return tuple(map(operator.add, a, b))

directions = {'W': (-1, 0), 'E': (1, 0), 'S': (0, -1), 'N': (0, 1)}
action = {1 : 'W', 2 : 'E', 3 : 'S', 4 : 'N'}


def xorHash(hash, point, player):
    print("In xorHash : ", hash)
    hash ^= goboardhash.HASH_CODE[point, player]
    return hash


class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        if self == Player.white:
            return Player.black 
        else:
            return Player.white

#directionVector_add((self.row, self.col),(1,1))
class Point(namedtuple('Point', 'row col')):
    def neighbors(self):
        neigh = []
        for i in range (1, len(directions) + 1):
            row, col = directionVector_add((self.row, self.col),(directions[action[i]]))
            neigh.append(Point(row, col))
            
        return neigh

    def __deepcopy__(self, memodict={}):
        return self





stone_bestMove = [Player.black, Player.white, Player.black, Player.white, Player.black, Player.white]

stone1_1 = [Point(3,3), Point(3,4), Point(3,2), Point(4,3), Point(2,3), Point(4,2)]
stone1_2 = [Point(3,3), Point(3,4), Point(3,2), Point(4,3), Point(2,3), Point(2,4)]

stone2_1 = [Point(3,3), Point(3,4), Point(3,2), Point(2,3), Point(4,3), Point(2,2)]
stone2_2 = [Point(3,3), Point(3,4), Point(3,2), Point(2,3), Point(4,3), Point(4,4)]

stone3_1 = [Point(3,3), Point(3,2), Point(3,4), Point(4,3), Point(2,3), Point(2,2)]
stone3_2 = [Point(3,3), Point(3,2), Point(3,4), Point(4,3), Point(2,3), Point(4,4)]

stone4_1 = [Point(3,3), Point(3,2), Point(3,4), Point(2,3), Point(4,3), Point(2,4)]
stone4_2 = [Point(3,3), Point(3,2), Point(3,4), Point(2,3), Point(4,3), Point(4,2)]

stone_pattern = [stone1_1, stone1_2, stone2_1, stone2_2, stone3_1, stone3_2, stone4_1, stone4_2]
#stone_point = [Point(2,4), Point(4,2), Point(4,4), Point(2,2), Point(4,4), Point(2,2), Point(4,2), Point(2,4)]



#print(stone_pattern)
#print("*" * 60)
#print(len(stone_pattern))


"""
# Generate the hash code for the board which may lead to dead stones.
hash_code = []
for j in range(0, len(stone_pattern)):
    hash = goboardhash.EMPTY_BOARD
    for i in range(0,6):
        print(stone_pattern[j])
        hash = xorHash(hash, stone_pattern[j][i], stone_bestMove[i])
        print(hash)
    print("*" * 60)   
    hash_code.append(hash)  

for j in range(0, len(stone_pattern)): 
    print("FINAL HASH: ", j, hash_code[j])
 
"""