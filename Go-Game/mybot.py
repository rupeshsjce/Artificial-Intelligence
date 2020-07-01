import time
import os
import numpy as np
import pprint

from alphabeta import *
from collections import namedtuple
from gohelper import Player, Point
from goboard import *

BOARD_SIZE = 5

stone_point = [Point(2,4), Point(4,2), Point(4,4), Point(2,2), Point(4,4), Point(2,2), Point(4,2), Point(2,4)]

H_CODE = [4127058818861117788,
             1112622235770286509,
             7342025575321657994,
             5748182797001150899,
             8771342974450464139,
             6024595183845131954,
             1382280374592497324,
             2672972738532058717]


H1_CODE = [5642301740485644788,
           7796708187844328524,
           5641366615823622734,
           7796903361110478838]
s_point = [Point(2,4), Point(4,4), Point(4,2), Point(2,2)]


H2_CODE = [8467627119832304022,
           4452962101753889526,
           4340509686934527100,
           6659110502266784677]
#s2_point = [Point(1,3), Point(5,3), Point(5,3), Point(1,3)]
s2_point = [Point(4,4), Point(2,4), Point(2,2), Point(4,2)]


def readInput(n, path="input.txt"):

    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board

def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
    	res = "PASS"
    else:
	    res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)    



def MY_TA_POSITION(action):
    row_MY = action[0]
    col_MY = action[1]
    
    row_TA = BOARD_SIZE - row_MY
    col_TA = col_MY - 1
    return (row_TA, col_TA)

def TA_MY_POSITION(action):
    row_TA = action[0]
    col_TA = action[1]
    
    row_MY = BOARD_SIZE - row_TA
    col_MY = col_TA + 1
    #print("TA_MY_POSITION : ", row_MY, col_MY)
    return (row_MY, col_MY)

def check_for_init_board(board):
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] != 0:
                return False
    return True

def opp_move(board1, board2, opp_stone):
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board1[i][j] != board2[i][j]:
                if board2[i][j] == opp_stone:
                   return Move.play(Point(i, j))
    return Move.pass_turn()



#______________________________________________________________________________
#evaluation function : just the stones difference on board
def stone_diff(game_state):
    black_stones = 0
    white_stones = 0
    for r in range(1, game_state.board.rows + 1):
        for c in range(1, game_state.board.cols + 1):
            p = Point(r, c)
            color = game_state.board.get(p)
            if color == Player.black:
                black_stones += 1
            elif color == Player.white:
                white_stones += 1
    #diff = black_stones - (white_stones + 2.5)
    diff = black_stones - white_stones
    if game_state.next_player == Player.black:
        return diff
    return -1 * diff


#______________________________________________________________________________
#START_POSITINAL_EVALUATION _FUNCTION
class Area:
    def __init__(self, territory_map):
        self.num_black_area_liberty = 0
        self.num_white_area_liberty = 0
        self.num_black_stones = 0
        self.num_white_stones = 0
        self.num_dame = 0
        self.dame_points = []
        for point, status in territory_map.items():
            if status == Player.black:
                self.num_black_stones += 1
            elif status == Player.white:
                self.num_white_stones += 1
            elif status == 'area_b':
                self.num_black_area_liberty += 1
            elif status == 'area_w':
                self.num_white_area_liberty += 1
            elif status == 'dame':
                self.num_dame += 1
                self.dame_points.append(point)

class BoardValue(namedtuple('BoardValue', 'b w')):
    @property
    def board_value(self):
        return self.b - self.w
        
def evaluate_area_liberty(board):

    status = {}
    for r in range(1, board.rows + 1):
        for c in range(1, board.cols + 1):
            p = Point(row=r, col=c)
            if p in status:
                continue
            stone = board.get(p)
            if stone is not None:
                status[p] = board.get(p)
            else:
                group, neighbors = _collect_region(p, board)
                if len(neighbors) == 1:
                    neighbor_stone = neighbors.pop()
                    stone_str = 'b' if neighbor_stone == Player.black else 'w'
                    fill_with = 'area_' + stone_str
                else:
                    fill_with = 'dame'
                for pos in group:
                    status[pos] = fill_with
    return Area(status)

def _collect_region(start_pos, board, visited=None):

    if visited is None:
        visited = {}
    if start_pos in visited:
        return [], set()
    all_points = [start_pos]
    all_borders = set()
    visited[start_pos] = True
    here = board.get(start_pos)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for delta_r, delta_c in deltas:
        next_p = Point(row=start_pos.row + delta_r, col=start_pos.col + delta_c)
        if not board.is_on_grid(next_p):
            continue
        neighbor = board.get(next_p)
        if neighbor == here:
            points, borders = _collect_region(next_p, board, visited)
            all_points += points
            all_borders |= borders
        else:
            all_borders.add(neighbor)
    return all_points, all_borders

# Linear function : diff(9 * territory + 10 * stones) means more weightage to stones on board
def compute_positional_result(game_state):
    area_liberties = evaluate_area_liberty(game_state.board)
    return BoardValue(
        2 * area_liberties.num_black_area_liberty + 10 * area_liberties.num_black_stones,
        2 * area_liberties.num_white_area_liberty + 10 * area_liberties.num_white_stones)

#evaluation function : 
#     check the unoccupied points enclosed by player groups + total number of player stones on board
def positional_diff(game_state):
    gameResult = compute_positional_result(game_state)
    diff = gameResult.board_value
    if game_state.next_player == Player.black:
        return diff
    return -1 * diff

#END_POSITIONAL_EVALUATION_FUNCTION
#______________________________________________________________________________

def my_bot_turn(game):
    
    start_time = time.time()
    
    bot_AB_0_CD = AlphaBetaAgent(0, stone_diff)
    bot_AB_0_TD = AlphaBetaAgent(0, positional_diff)
    
    
    bot_AB_1_CD = AlphaBetaAgent(1, stone_diff)
    bot_AB_1_TD = AlphaBetaAgent(1, positional_diff)
    
    bot_AB_2_CD = AlphaBetaAgent(2, stone_diff)
    bot_AB_2_TD = AlphaBetaAgent(2, positional_diff)
    
    bot_AB_3_CD = AlphaBetaAgent(3, stone_diff)
    bot_AB_3_TD = AlphaBetaAgent(3, positional_diff)
    
    print("[IMP] Player and completed moves : ", game.next_player, game.board.moves)
    move_selected = False
    if game.next_player == Player.black:
        # I am playing as black (second last move will be mine)
        # game.board.moves  will give how many moves are already played
        # for game.board.moves = 1, it won't land here
        # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        # B W B W B W B W B  W  B  W  B  W  B  W  B  W  B  W  B  W  B  W
        
        index = game.board.moves + 1                   #My move index/postion is completed moves + 1
      
        if index == 3:                                 # for move #2 (3,2) or (3,4) OR Play one of the diagonal (2,2)(2,4)(4,2)(4,4)
            move = Move.play(Point(3,2))
            if not game.board.get(move.point) is None: # Means point is occupied
                move = Move.play(Point(3,4))
            move_selected = True
            #bot = bot_AB_2_TD
        elif index == 5:                       #3
            
            
            if game.board._hash in H1_CODE:
               print("GAME BOARD IS PRESENT IN H1_CODE ", game.board._hash)
               index = H1_CODE.index(game.board._hash)
               move = Move.play(s_point[index])
               if not game.board.get(move.point) is None:
                   bot = bot_AB_2_CD
                   move_selected = False
               else:
                   print("PLAYED ", move.point)
                   move_selected = True
            else:
               #bot = bot_AB_2_CD
                move_selected = True
                move = Move.play(Point(2,3))
                if not game.board.get(move.point) is None:
                    move = Move.play(Point(4,3))
                    if not game.board.get(move.point) is None:
                        move = Move.play(Point(3,4))
                        if not game.board.get(move.point) is None:
                            move = Move.play(Point(3,2))
                            if not game.board.get(move.point) is None:
                                # None of the above points are available, so use bot to pick up the move
                                bot = bot_AB_2_CD
                                move_selected = False
            
    
            #"""
            #move_selected = True
            #move = Move.play(Point(2,3))
            #if not game.board.get(move.point) is None:
            #    move = Move.play(Point(4,3))
            #    if not game.board.get(move.point) is None:
            #        move = Move.play(Point(3,4))
            #        if not game.board.get(move.point) is None:
            #            move = Move.play(Point(3,2))
            #            if not game.board.get(move.point) is None:
            #                # None of the above points are available, so use bot to pick up the move
            #                bot = bot_AB_2_CD
            #                move_selected = False
            #"""
                            
                            
        elif index == 7: #4 
            
            if game.board._hash in H2_CODE:
                print("GAME BOARD IS PRESENT IN H2_HASH ", game.board._hash)
                index = H2_CODE.index(game.board._hash)
                move = Move.play(s2_point[index])
                if game.board.get(move.point) is None:
                    print("PLAYED from H2: ", move.point)
                    move_selected = True
                else:
                    bot = bot_AB_2_CD
                    move_selected = False            
            
            elif game.board._hash in H_CODE:
                print("GAME BOARD IS PRESENT IN HASH CALCULATED", game.board._hash)
                index = H_CODE.index(game.board._hash)
                move = Move.play(stone_point[index])
                if not game.board.get(move.point) is None:
                    bot = bot_AB_2_CD
                    move_selected = False
                else:
                    print("PLAYED: ", move.point)
                    move_selected = True
            else:
                bot = bot_AB_2_CD
                            
        elif index == 9:         # #5      ## Let's see the move 4 for the dead-corner.    
            bot = bot_AB_3_CD 
        elif index in [17,19,21,23]:           # #9 #10 #11 #12
            print("last 4 moves")
            if index == 23:
                bot = bot_AB_1_CD
            else:    
                bot = bot_AB_3_CD   
        #elif index == 21:                     # depth = 3 eval: CD for #11
        #    bot = bot_AB_3_CD
        #    print("MOVE #11")
        #elif index == 23:
        #    print("MOVE #12")
        #    bot = bot_AB_1_CD                 # depth = 1 eval: CD for #12
        #elif index == 19:
        #    print("MOVE #10 but CD")          # depth = 3 eval: CD for #10
        #    bot = bot_AB_3_CD     
        else:                                  # depth = 3 eval: TD for #6 to #8
            print("MOVE #6 to #8")
            bot = bot_AB_3_CD
            
    if game.next_player == Player.white:
        # I am playing as white (last move will be mine)
        # game.board.moves  will give how many moves are already played
        # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        # B W B W B W B W B  W  B  W  B  W  B  W  B  W  B  W  B  W  B  W
        
        index = game.board.moves + 1 #My move index/postion is completed moves + 1
        
        if index == 2 or index == 4:          # depth = 1 eval: TD for #1 and #2
            if index == 2:                    # for #1 (3,3) or (4,3)
                move = Move.play(Point(3,3))
                if not game.board.get(move.point) is None:
                    move = Move.play(Point(4,3))
                move_selected = True                    
            else:
                move_selected = True
                #(3,2) or (3,4) or random fo #2
                move = Move.play(Point(3,2))
                if not game.board.get(move.point) is None:
                     move = Move.play(Point(3,4))
                     if not game.board.get(move.point) is None:
                         bot = bot_AB_2_CD # both points are occupied 
                         move_selected = False   
  
                #bot = bot_AB_2_CD
        elif index == 6 or index == 8:        # depth = 2 eval: TD for #3 #4
            bot = bot_AB_2_CD
            print("MOVE #3 and #4")
        elif index == 20:                     # depth = 2 eva: CD for #10
            bot = bot_AB_2_CD
            print("MOVE #10")
        elif index == 22:                     # depth = 2 eval: CD* for #11
            bot =  bot_AB_2_CD
            print("MOVE #11")            
        elif index == 24:                     # depth = 0 eval: CD for #12
            bot = bot_AB_0_CD
            print("MOVE #12")
        else:
            print("MOVE #5 to #9")
            bot = bot_AB_3_TD                 # depth = 3 eval: TD for #5 to #9

    
    #bot = AlphaBetaAgent(2, stone_diff)
    if not move_selected: 
       move = bot.pick_move(game)
    
    if time.time()- start_time > 10:
        print("[MYEROR] Execution time for my bot is more than 10s :", time.time()- start_time)
        
    return move
    
    """
    #Testing plain alpha-beta
    bot_AB_2_CD = AlphaBetaAgent(2, stone_diff)
    bot = bot_AB_2_CD
    move = bot.pick_move(game)
    return move
    """


def play_my_turn(is_first_move, piece_type, previous_board, board, game):
    
    #bot = AlphaBetaAgent(2, stone_diff)
    
    #STEP-1
    """
    Find the opposition last move (compare previous_board and board) for the 
    opposition Stone as current board now have many opposition capture and so
    huge difference in board co-ord. So, just find the opposition last move.
    It will be pass if both board is same or (row,col)
    """
    move = opp_move(previous_board, board, game.next_player.other.value) # This move co-ord is with TAs Board
    #print("Point : ", move.point)
    # Save the move to the file sequence_moves.txt
    f = open("sequence_moves.txt", "a")
    if move.is_pass:
        row, col = 0, 0
    else:
        action = move.point.row, move.point.col # TA's Board co-ord
        action = TA_MY_POSITION(action) # MY Board co-ord
        row, col = action[0], action[1]
    if is_first_move:  
        f.write(str(game.next_player.other.value) + " " + str(row) + " " + str(col))
    else:    
        f.write('\n'+ str(game.next_player.other.value) + " " + str(row) + " " + str(col))    
    f.close()
    
    
    #STEP-2
    """
    Open the file: sequence_moves.txt to read the moves and place it on board.
    """
    with open('sequence_moves.txt', "r") as f:
         array = [[int(x) for x in line.split()] for line in f]
    ##print(array)
    f.close()
    
    # Play all the saved moves.
    for move in array:
        #print(move, move[0], move[1], move[2])
        player, row, col = move[0], move[1], move[2]
        game.next_player = Player(player)
        if (row == 0 and col == 0):
            #This is a pass by player(1 or 2)
            move = Move.pass_turn()
            game = game.apply_move(move)
            game.board.moves += 1
            
        else:    
           #game.board.place_chess(Player(player), Point(row, col))
           move = Move.play(Point(row, col))
           game = game.apply_move(move)
           game.board.moves += 1
           #print_board(game.board)
           
        #print("MOVE : ", game.board.moves)
 
    #STEP-3
    """
    Now board is ready and it is my turn to make move, so call the agent
    Make a function to pass (game, bot) to tweak the performance.
    """
    #print(game.next_player, game.next_player.value)
    move = my_bot_turn(game)
    #move = bot.pick_move(game)
    print("Point : ", move, move.point, time.time() - start_time)
    
    #STEP-4
    """
    Save my latest move to the file sequence_moves.txt
    """
    f = open("sequence_moves.txt", "a")
    if move.is_pass:
        row, col = 0, 0
        action = "PASS"
    else:
        row, col = move.point.row, move.point.col
        action = (row, col)
    f.write('\n'+ str(game.next_player.value) + " " + str(row) + " " + str(col))    
    f.close()
    
    #STEP-5
    """
    write (row,col) or PASS to the output.txt
    """
    if action != "PASS":
        action = MY_TA_POSITION(action)
    writeOutput(action)
    
    """
    # Below lines will not be needed.
    game = game.apply_move(move)
    game.board.moves += 1
    print("MOVE : ", game.board.moves)
    """
    #game = game.apply_move(move) # NOT NEEDED AS SUCH
    #game.board.moves += 1        # NOT NEEDED AS SUCH
    #print_board(game.board)

#end


if __name__ == '__main__':        
    start_time = time.time()    
    #bot = AlphaBetaAgent(2, stone_diff)



    # Start your new code here.
    piece_type, previous_board, board = readInput(BOARD_SIZE)
    
    #print(piece_type, previous_board, board)
    is_first_move = check_for_init_board(previous_board)
    #print("MY MOVE IS FIRST? : ", is_first_move)
    if (is_first_move):
        #remove the sequence_moves.txt [or open the file in "w" mode]
        if os.path.exists("sequence_moves.txt"):
            #print("Deleting the sequence_moves.txt")
            os.remove("sequence_moves.txt") #TODO
        else:
            #print("No file to delete, sequence_moves.txt")
            pass
    else:
        #print("This is not the first move")
        pass



    start_time = time.time()
    game = GoState.new_game(BOARD_SIZE)
    game.next_player = Player(piece_type) # This stone is MY BOT
    #print_board(game.board)        
    
    if not is_first_move:
        # This is my #2 ... (2nd (including) move ownwards)
        play_my_turn(is_first_move, piece_type, previous_board, board, game)
    else:
        # This is MY FIRST MOVE
        f = open("sequence_moves.txt", "w")
        if piece_type == 1:
            """
            SPECIAL-CASE-BLACK
            Play (3,3). Means create-write (1 3 3) to the sequence_moves.txt and 
            write (3,3) to output.txt
            """
            f.write(str(piece_type) + " " + str(3) + " " + str(3))
            f.close()
            
            #write (3 3) to the output.txt but in TAs board format
            action = (3, 3)
            action = MY_TA_POSITION(action)
            writeOutput(action)
        else:  #piece_type == 2:
            f.close()
            play_my_turn(is_first_move, piece_type, previous_board, board, game)

    print("Execution time for my bot : ",time.time() - start_time)