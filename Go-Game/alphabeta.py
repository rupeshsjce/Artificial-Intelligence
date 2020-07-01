import numpy as np
import random
from gohelper import Player

MAX_SCORE = np.inf
MIN_SCORE = -np.inf


def UTILITY_VALUE(game_state, eval_fn):
    eval_fn_val = eval_fn(game_state)
    return eval_fn_val 

def returnMaxMinVal(game_state):                                  
    winner , winner_margin = game_state.winner()
    if winner == game_state.next_player:      
        return MAX_SCORE                                   
    else:                                                  
        return MIN_SCORE    


def returnBestBoardValue(Player, best_so_far, v_white, v_black):
    if Player.white:
        if best_so_far > v_white:
            v_white = best_so_far
        outcome_for_black = -1 * best_so_far
        if outcome_for_black < v_black:
            return best_so_far
    if Player.black:
        if best_so_far > v_black: 
            v_black = best_so_far
        outcome_for_white = -1 * best_so_far
        if outcome_for_white < v_white:
            return best_so_far
            

def ALPHA_BETA_SEARCH(game_state, max_d, v_black, v_white, eval_fn):
    #If game is over then return the MAX/MIN Value.
    if game_state.is_over():                                  
       return returnMaxMinVal(game_state) 
   
    #If depth = 0, evaluate the board with evaluation function.
    if max_d == 0:
        return UTILITY_VALUE(game_state, eval_fn)                                         
                                    

    best_so_far = MIN_SCORE
    for move in game_state.legal_moves():            
        next_state = game_state.apply_move(move)     
        min_val = ALPHA_BETA_SEARCH(next_state, max_d - 1, v_black, v_white, eval_fn) 
        # my bot best value is reverse of opposition best value                                          
        max_val = -1 * min_val

        if max_val > best_so_far:                           
            best_so_far = max_val

        """
        outcome_for_black = 0
        outcome_for_white = 0
        player = [[1],[2]]
        player[Player.black] = [v_black, outcome_for_black]
        player[Player.white] = [v_white, outcome_for_white]
       
        if best_so_far > player[game_state.next_player][0]:
            player[game_state.next_player][0] = best_so_far
        player[game_state.next_player.other][1] = -1 * best_so_far
        if player[game_state.next_player.other][1] < player[game_state.next_player.other][0]:
            return best_so_far
            
        
        """

        if game_state.next_player == Player.white:
            #returnBestBoardValue(Player, best_so_far, best_white, best_black)
            
            if best_so_far > v_white:                       
                v_white = best_so_far                       
            outcome_for_black = -1 * best_so_far               
            if outcome_for_black < v_black:                 
                return best_so_far                             
            
            
        elif game_state.next_player == Player.black:
            #returnBestBoardValue(Player, best_so_far, best_white, best_black)
            
            if best_so_far > v_black:                       
                v_black = best_so_far                       
            outcome_for_white = -1 * best_so_far               
            if outcome_for_white < v_white:                 
                return best_so_far
           

    return best_so_far



"""
AlphaBetaAgent is with max_depth and evaluation function.
"""
class AlphaBetaAgent():
    def __init__(self, max_d, eval_fn):
        self.max_d = max_d
        self.current_depth = 0
        self.eval_fn = eval_fn
        

    def pick_move(self, game_state):
        # list to contain the best moves for the current game_state as per eval fn.
        best_moves = []
        # best values for black and white
        v_black = -np.inf
        v_white = -np.inf
        best_score = None
        
        # Loop over all legal moves.
        for move in game_state.legal_moves():
            next_state = game_state.apply_move(move)
            min_val = ALPHA_BETA_SEARCH(next_state, self.max_d,v_black, v_white, self.eval_fn)
            #Make the best move for MAX assuming that MIN always replies with the best move for MIN
            max_val = -1 * min_val
            if (not best_moves) or max_val > best_score:
                best_score = max_val
                best_moves = [move]
                if game_state.next_player == Player.black:
                    v_black = best_score
                elif game_state.next_player == Player.white:
                    v_white = best_score
            elif max_val == best_score:
                best_moves.append(move)
            
        if len(best_moves) > 1: #and game_state.board.moves == 3:    
            b_best_moves = []    
            for m in best_moves:
              if m.is_pass:
                  continue
              elif 2<= m.point.row <= 4 and 2 <= m.point.col <=4:
                  b_best_moves.append(m)
              else:
                  continue
              
            if len(b_best_moves) >= 1:
                 return random.choice(b_best_moves)    
                                  

                
        return random.choice(best_moves)

