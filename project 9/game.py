import random
import copy
class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def run_challenge_test(self):
        """ Set to True if you would like to run gradescope against the challenge AI!
        Leave as False if you would like to run the gradescope tests faster for debugging.
        You can still get full credit with this set to False
        """ 
        return True

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        #drop_phase = True   # TODO: detect drop phase
        pieces_on_board = sum(row.count('b') + row.count('r') for row in state)
        drop_phase = pieces_on_board < 8
        # move = []

        if not drop_phase:
            # Move phase: Find the best move using minimax
            best_value = float('-inf')
            best_move = None
            my_locations = []
            for r in range(5):
                for c in range(5):
                    if state[r][c] == self.my_piece:
                        my_locations.append((r, c))

            for r_src, c_src in my_locations:
                for r_dest in range(max(0, r_src - 1), min(5, r_src + 2)):
                    for c_dest in range(max(0, c_src - 1), min(5, c_src + 2)):
                        if state[r_dest][c_dest] == ' ' and (r_dest, c_dest) != (r_src, c_src):
                            next_state = copy.deepcopy(state)
                            next_state[r_src][c_src] = ' '
                            next_state[r_dest][c_dest] = self.my_piece
                            value, _ = self.min_value(next_state, 1, float('-inf'), float('inf'))
                            if value > best_value:
                                best_value = value
                                best_move = [(r_dest, c_dest), (r_src, c_src)]

            if best_move:
                return best_move
            else:
                my_locations = [(r, c) for r in range(5) for c in range(5) if state[r][c] == self.my_piece]
                for r_src, c_src in random.sample(my_locations, len(my_locations)):
                    for r_dest in range(max(0, r_src - 1), min(5, r_src + 2)):
                        for c_dest in range(max(0, c_src - 1), min(5, c_src + 2)):
                            if state[r_dest][c_dest] == ' ' and (r_dest, c_dest) != (r_src, c_src):
                                return [(r_dest, c_dest), (r_src, c_src)]

        else:
            # Drop phase: Find the best drop location using minimax
            best_value = float('-inf')
            best_drop = None
            for r in range(5):
                for c in range(5):
                    if state[r][c] == ' ':
                        next_state = copy.deepcopy(state)
                        next_state[r][c] = self.my_piece
                        value, _ = self.min_value(next_state, 1, float('-inf'), float('inf'))
                        if value > best_value:
                            best_value = value
                            best_drop = (r, c)

            if best_drop:
                return [best_drop]

    def succ(self, state):
        successors = []
        drop_phase = True
        r, b = 0, 0

        for row in range(5):
            for col in range(5):
                if(state[row][col]) == 'r':
                    r += 1
                if(state[row][col]) == 'b':
                    b += 1
        if(r == 4 and b == 4):
            drop_phase = False

        if drop_phase:
            for row in range(5):
                for col in range(5):
                    if state[row][col] == ' ':
                        state_copy = copy.deepcopy(state)
                        state_copy[row][col] = self.my_piece
                        successors.append(state_copy)
        else:
            for row in range(5):
                for col in range(5):
                    current_piece = state[row][col]
                    if current_piece == self.my_piece:
                        next_r = [row - 1, row, row + 1]
                        next_c = [col - 1, col, col + 1]
                        for row_next in next_r:
                            for col_next in next_c:
                                if row_next >= 0 and row_next < 5 and col_next >= 0 and col_next < 5:
                                    if state[row_next][col_next] == ' ':
                                        state_copy = copy.deepcopy(state)
                                        state_copy[row][col] = ' '
                                        state_copy[row_next][col_next] = self.my_piece
                                        successors.append(state_copy)
        

        return successors

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2] == state[row+3][col+3]:
                    return 1 if state[row][col] == self.my_piece else -1
                
        # TODO: check / diagonal wins
        for row in range(2):
            for col in range(3, 5):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col-1] == state[row+2][col-2] == state[row+3][col-3]:
                    return 1 if state[row][col] == self.my_piece else -1
                
        # TODO: check box wins
        for row in range(4):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row][col+1] == state[row+1][col] == state[row+1][col+1]:
                    return 1 if state[row][col] == self.my_piece else -1

        return 0 # no winner yet
    

    def heuristic_game_value(self, state):
            
            winner = self.game_value(state)
            if winner == 1 or winner == -1:
                return winner

            my_score = 0
            op_score = 0

            for row in range(5):
                for new_col in range(2):
                    curr_my_score = 0
                    curr_op_score = 0
                    for i in range(new_col, new_col + 4):
                        if state[row][i] == self.my_piece:
                            curr_my_score += 1
                        elif state[row][i] == self.opp:
                            curr_op_score += 1
                    my_score = max(my_score, curr_my_score)
                    op_score = max(op_score, curr_op_score)
                

            for row in range(2):
                for col in range(2):
                    curr_my_score = 0
                    curr_op_score = 0
                    for new in range(4):
                        if state[row + new][col + new] == self.my_piece:
                            curr_my_score += 1
                        elif state[row + new][col + new] == self.opp:
                            curr_op_score += 1
                        my_score = max(my_score, curr_my_score)
                        opp_score = max(op_score, curr_op_score)
            for row in range(2):
                for col in range(3, 5):
                    curr_my_score = 0
                    curr_op_score = 0
                    for new in range(4):
                        if state[row + new][col - new] == self.my_piece:
                            curr_my_score += 1
                        elif state[row + new][col - new] == self.opp:
                            curr_op_score += 1
                        my_score = max(my_score, curr_my_score)
                        op_score = max(op_score, curr_op_score)

            for row in range(4):
                for col in range(4):
                    box = [(row,col), (row,col+1), (row+1,col), (row+1, col+1)]
                    curr_my_score = 0
                    curr_op_score = 0
                    for r,c in box:
                        if state[r][c] == self.my_piece:
                            curr_my_score += 1
                        elif state[r][c] == self.opp:
                            curr_op_score += 1
                        my_score = max(my_score, curr_my_score)
                        op_score = max(opp_score, curr_op_score)

            if my_score > op_score:
                return my_score/4
            elif my_score < op_score:
                return my_score/-4
            else:
                return 0
    
    def min_value(self, state, depth, a, b):
        val = self.game_value(state)
        if val == 1 or val == -1:
            return val, state
        if depth == 3:
            return self.heuristic_game_value(state), state
    
        min_state = state
        success = self.succ(state)
        for succ in success:
            succ_b, _ = self.max_value(succ, depth+1, a, b)
            if succ_b < b:
                b = succ_b
                min_state = succ
            if a >= b:
                break
        return b, min_state

    def max_value(self, state, depth, a, b):
        val = self.game_value(state)
        if val == 1 or val == -1:
            return val, state
        if depth == 3:
            return self.heuristic_game_value(state), state
    
        max_state = state
        success = self.succ(state)
        for succ in success:
            succ_a, _ = self.min_value(succ, depth+1, a, b)
            if succ_a > a:
                a = succ_a
                max_state = succ
            if a >= b:
                break
        return a, max_state


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
