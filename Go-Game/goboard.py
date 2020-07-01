import copy
from gohelper import Player, Point
import goboardhash
BOARD_SIZE = 5

def compute_game_result(game_state):
    p1_stats = [0, 0, 0]
    
    for r in range(1, BOARD_SIZE + 1):
        for c in range(1, BOARD_SIZE + 1):
            p = Point(row=r, col=c)
            #print(p)
            if game_state.board.get(p) is not None:
                player = game_state.board.get(p).value
                p1_stats[player] = p1_stats[player] + 1

    return p1_stats
 



def updateGroupsHash(board, player, point, liberties, ally_stones, opp_ally_stones):
    
    new_connectedStones = ConnectedStones(player, [point], liberties)
    new_connectedStones = ConnectedStones(player, [point], liberties)

    for same_color_string in ally_stones:
        new_connectedStones = new_connectedStones.combineGroupStones(same_color_string)
    for new_string_point in new_connectedStones.stones:
        board._grid[new_string_point] = new_connectedStones

    board._hash ^= goboardhash.HASH_CODE[point, player]

    for other_color_string in opp_ally_stones:
        replacement = other_color_string.add_remove_liberty(point, False)
        if replacement.num_liberties:
            board._replace_connectedStones(other_color_string.add_remove_liberty(point, False))
        else:
            board._remove_connectedStones(other_color_string)
                

class ConnectedStones:
    def __init__(self, color, stones, liberties):
        self.color = color
        self.is_dead = False
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)

    def add_remove_liberty(self, point, add_remove):
        if add_remove:
            new_liberties = self.liberties | set([point])
        else:
             new_liberties = self.liberties - set([point])
        return ConnectedStones(self.color, self.stones, new_liberties)

    @property
    def num_liberties(self):
        n_liberties = len(self.liberties)
        return n_liberties

    #Check if the group of stones are dead.
    def dead_GroupStones(self):
        if self.liberties == 0:
            self.is_dead = True
            
    """Return a new ConnectedStones containing all stones in both groups."""
    def combineGroupStones(self, string):
        combined_stones = self.stones | string.stones
        final_liberties = (self.liberties | string.liberties) - combined_stones
        connectedStones = ConnectedStones(self.color, combined_stones, final_liberties)
        return connectedStones
            
            
    def __eq__(self, other):
        return isinstance(other, ConnectedStones) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties
            
    # Own deep copy implementation, __deepcopy__() implementation needs 
    # to make a deep copy of a component (liberties information).
    def __deepcopy__(self, memodict={}):
        connectedStones = ConnectedStones(self.color, self.stones, copy.deepcopy(self.liberties))
        return connectedStones



class GoBoard:
    def __init__(self, rows, cols, moves = 0):
        self.rows = rows
        self.cols = cols
        self.moves = moves # Number of moves played till now.
        self._grid = {}
        #Init the board with EMPTY BOARD HASH
        self._hash = goboardhash.EMPTY_BOARD
        self.komi = rows/2 # Komi rule
        self.verbose = True
        self.max_move = rows * rows - 1 # The max moves allowed for a Go game


    def place_chess(self, player, point):
        
        opp_ally_stones = []
        ally_stones = []
        liberties = []
        
        for neighbor in point.neighbors():
            # neighbor is out of grid.
            if not self.is_on_grid(neighbor):
                continue
            
            # Get ConnectedStones for this neighbor
            neighbor_connectedStones = self._grid.get(neighbor)
            
            if neighbor_connectedStones is None:
                liberties.append(neighbor)
            elif neighbor_connectedStones.color == player:
                if neighbor_connectedStones not in ally_stones:
                    ally_stones.append(neighbor_connectedStones)
            else:
                if neighbor_connectedStones not in opp_ally_stones:
                    opp_ally_stones.append(neighbor_connectedStones)
         
        updateGroupsHash(self, player, point, liberties, ally_stones, opp_ally_stones)  

    def _replace_connectedStones(self, new_connectedStones):  
        for point in new_connectedStones.stones:
            self._grid[point] = new_connectedStones

    def _remove_connectedStones(self, connectedStones):
        for point in connectedStones.stones:
            for neighbor in point.neighbors():
                neighbor_connectedStones = self._grid.get(neighbor)
                if neighbor_connectedStones is None:
                    continue
                if neighbor_connectedStones is not connectedStones:
                    self._replace_connectedStones(neighbor_connectedStones.add_remove_liberty(point, True))
            self._grid[point] = None

            self._hash ^= goboardhash.HASH_CODE[point, connectedStones.color]  


    def is_on_grid(self, point):
        return 1 <= point.row <= self.rows and \
            1 <= point.col <= self.cols

    def get(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color


    def __eq__(self, other):
        return isinstance(other, GoBoard) and \
            self.rows == other.rows and \
            self.cols == other.cols and \
            self._hash() == other._hash()

    def __deepcopy__(self, memodict={}):
        copied = GoBoard(self.rows, self.cols, self.moves)
        copied._grid = copy.copy(self._grid)
        copied._hash = self._hash
        return copied


    def goboardhash_hash(self):
        return self._hash
    
    def getStoneGroups(self, point):
        string = self._grid.get(point)
        if string is None:
            return None
        return string    


class GoState:
    def __init__(self, board, next_player, previous, move):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        if self.previous_state is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous.previous_states |
                {(previous.next_player, previous.board.goboardhash_hash())})
        self.last_move = move


    def apply_move(self, move):
        """Return the new GoState after applying the move."""
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place_chess(self.next_player, move.point)
        else:
            next_board = self.board
        return GoState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = GoBoard(*board_size)
        return GoState(board, Player.black, None, None)

    def is_suicide(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_chess(player, move.point)
        new_string = next_board.getStoneGroups(move.point)
        return new_string.num_liberties == 0

    @property
    def situation(self):
        return (self.next_player, self.board)

    # Check special case: repeat placement causing the repeat board state (KO rule)
    def violate_ko(self, player, move):
        if not move.is_play:
            return False
        next_board = copy.deepcopy(self.board)
        next_board.place_chess(player, move.point)
        next_situation = (player.other, next_board.goboardhash_hash())
        return next_situation in self.previous_states

    # Return all the legal moves.
    def legal_moves(self):
        l_moves = []
        for row in range(1, BOARD_SIZE + 1):
            for col in range(1, BOARD_SIZE + 1):
                move = Move.play(Point(row, col))
                if self.is_valid_move(move):
                    l_moves.append(move)
        l_moves.append(Move.pass_turn())

        return l_moves

    
    #Check whether the move is valid or not. 
    def is_valid_move(self, move):
        if self.is_over():
            return False
        if move.is_pass:
            return True
        return (
            self.board.get(move.point) is None and
            not self.is_suicide(self.next_player, move) and
            not self.violate_ko(self.next_player, move))
    
    def is_over(self):
        if self.board.moves >= self.board.max_move:
            return True
        if self.last_move is None:
            return False
        opp_last_move = self.previous_state.last_move
        if opp_last_move is None:
            return False
        """
        If the last move and opp last move is PASS then Game is over
        """
        return self.last_move.is_pass and opp_last_move.is_pass
    
    
    def winner(self):
        if not self.is_over():
            #return None
            print("GAME IS NOT OVER")
            return 0, 0
        
        p1_stats = compute_game_result(self)
        if p1_stats[1] > p1_stats[2] + self.board.komi:
            winner = Player(1)
            winner_margin = abs(p1_stats[1] - (p1_stats[2] + self.board.komi))
        elif(p1_stats[1] == p1_stats[2] + self.board.komi):
            print("MATCH DRAWN, WHICH IS NOT POSSIBLE")
            winner = Player(0)
            winner_margin = 0
        else:
            winner = Player(2)
            winner_margin = abs(p1_stats[1] - (p1_stats[2] + self.board.komi))
            
            
        return winner, winner_margin
    

class Move:
    def __init__(self, point=None, is_pass=False):
        self.point = point
        self.is_play = (self.point is not None)
        self.is_pass = is_pass
        self.is_selected = False

    @classmethod
    def play(cls, point):
        return Move(point=point)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass=True)
    
    @classmethod
    def ply_selected(cls):
        return Move(is_selected = True)
