import sqlite3
import random
import numpy as np
import chess
import chess.engine
import chess.pgn
import chess.svg
import sklearn
from sklearn.linear_model import LogisticRegression
import time
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

pgna = open("pgn/Modern.pgn")
pgnb = open("pgn/QG-Albin.pgn")
pgnc = open("pgn/GiuocoPiano.pgn")
k_vs_d = ['rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2', 'rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3', 'rnbqkb1r/pppp1ppp/4pn2/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3', 'rnbqkb1r/pppp1ppp/4pn2/8/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 1 3', 'rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4', 'rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 1 4', 'rnbqkb1r/pp3ppp/4pn2/2pp4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5', 'rnbqkb1r/pp3ppp/4pn2/2pP4/3P4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 0 5', 'rnbqkb1r/pp3ppp/4pn2/3P4/3p4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6', 'rnbqkb1r/pp3ppp/4pn2/3P4/3Q4/2N2N2/PP2PPPP/R1B1KB1R b KQkq - 0 6', 'rnbqkb1r/pp3ppp/5n2/3p4/3Q4/2N2N2/PP2PPPP/R1B1KB1R w KQkq - 0 7', 'rnbqkb1r/pp3ppp/5n2/3p2B1/3Q4/2N2N2/PP2PPPP/R3KB1R b KQkq - 1 7', 'rnbqk2r/pp2bppp/5n2/3p2B1/3Q4/2N2N2/PP2PPPP/R3KB1R w KQkq - 2 8', 'rnbqk2r/pp2bppp/5n2/3p2B1/3Q4/2N1PN2/PP3PPP/R3KB1R b KQkq - 0 8', 'rnbq1rk1/pp2bppp/5n2/3p2B1/3Q4/2N1PN2/PP3PPP/R3KB1R w KQ - 1 9', 'rnbq1rk1/pp2bppp/5n2/3p2B1/3Q4/2N1PN2/PP3PPP/3RKB1R b K - 2 9', 'r1bq1rk1/pp2bppp/2n2n2/3p2B1/3Q4/2N1PN2/PP3PPP/3RKB1R w K - 3 10', 'r1bq1rk1/pp2bppp/2n2n2/3p2B1/Q7/2N1PN2/PP3PPP/3RKB1R b K - 4 10', 'r2q1rk1/pp2bppp/2n1bn2/3p2B1/Q7/2N1PN2/PP3PPP/3RKB1R w K - 5 11', 'r2q1rk1/pp2bppp/2n1bn2/1B1p2B1/Q7/2N1PN2/PP3PPP/3RK2R b K - 6 11', 'r4rk1/pp2bppp/1qn1bn2/1B1p2B1/Q7/2N1PN2/PP3PPP/3RK2R w K - 7 12', 'r4rk1/pp2bppp/1qn1bB2/1B1p4/Q7/2N1PN2/PP3PPP/3RK2R b K - 0 12', 'r4rk1/pp3ppp/1qn1bb2/1B1p4/Q7/2N1PN2/PP3PPP/3RK2R w K - 0 13', 'r4rk1/pp3ppp/1qn1bb2/1B1N4/Q7/4PN2/PP3PPP/3RK2R b K - 0 13', 'r4rk1/pp3ppp/1qn2b2/1B1b4/Q7/4PN2/PP3PPP/3RK2R w K - 0 14', 'r4rk1/pp3ppp/1qn2b2/1B1R4/Q7/4PN2/PP3PPP/4K2R b K - 0 14', 'r4rk1/pp3ppp/1qn5/1B1R4/Q7/4PN2/Pb3PPP/4K2R w K - 0 15', 'r4rk1/pp3ppp/1qn5/1B1R4/Q7/4PN2/Pb2KPPP/7R b - - 1 15', 'r4rk1/pp3ppp/1qn2b2/1B1R4/Q7/4PN2/P3KPPP/7R w - - 2 16', 'r4rk1/pp3ppp/1qn2b2/1B1R4/Q7/4PN2/P3KPPP/3R4 b - - 3 16', '2r2rk1/pp3ppp/1qn2b2/1B1R4/Q7/4PN2/P3KPPP/3R4 w - - 4 17', '2r2rk1/pp3ppp/1qn2b2/3R4/Q1B5/4PN2/P3KPPP/3R4 b - - 5 17', '2r2rk1/pp3ppp/2n2b2/3R4/QqB5/4PN2/P3KPPP/3R4 w - - 6 18', '2r2rk1/pp3ppp/2n2b2/3R4/1qB5/1Q2PN2/P3KPPP/3R4 b - - 7 18', '2r2rk1/pp3ppp/2n2b2/3R4/2B5/1q2PN2/P3KPPP/3R4 w - - 0 19', '2r2rk1/pp3ppp/2n2b2/3R4/8/1B2PN2/P3KPPP/3R4 b - - 0 19', '1nr2rk1/pp3ppp/5b2/3R4/8/1B2PN2/P3KPPP/3R4 w - - 1 20', '1nr2rk1/pp3ppp/5b2/3R4/6P1/1B2PN2/P3KP1P/3R4 b - - 0 20', '1nr2rk1/pp3pp1/5b1p/3R4/6P1/1B2PN2/P3KP1P/3R4 w - - 0 21', '1nr2rk1/pp3pp1/5b1p/3R4/6PP/1B2PN2/P3KP2/3R4 b - - 0 21', '1nr2rk1/pp3p2/5bpp/3R4/6PP/1B2PN2/P3KP2/3R4 w - - 0 22', '1nr2rk1/pp3p2/5bpp/3R2P1/7P/1B2PN2/P3KP2/3R4 b - - 0 22', '1nr2rk1/pp3p2/5bp1/3R2p1/7P/1B2PN2/P3KP2/3R4 w - - 0 23']
bishop = 0
pawn = 0
knight = 0
rook = 0
queen = 0

#engine = chess.engine.SimpleEngine.popen_uci("/usr/bin/stockfish")

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

conn = sqlite3.connect('data.db')
c = conn.cursor()
move = {}

def get_positions(num_games, pgn, fen_arr, result_arr):
    for i in range(num_games):
        try:
            game = chess.pgn.read_game(pgn)

            board = game.board()
            if game.headers["Result"] == '1-0':
                result = 1
            elif game.headers["Result"] == '0-1':
                result = -1
            else:
                continue
            num_moves = 0

            for k in game.mainline_moves():
                board.push(k)
                num_moves += 1

            board = game.board()

            count = 1
            for j in game.mainline_moves():
                if count % 2 == 1:
                    if num_moves-count in [5, 7, 9, 11, 13]:
                        if num_moves-count in move:
                            move[num_moves-count].append(len(fen_arr))
                        else:
                            move[num_moves-count] = [len(fen_arr)]
                board.push(j)
                if count % 2 == 1:
                    fen = board.board_fen()
                    fen_arr.append(fen)
                    result_arr.append(result)
                count += 1
        except:
            pass
    return fen_arr, result_arr

def get_fen(num_games):
    fen_arr = []
    np.array(fen_arr)
    result_arr = []
    np.array(result_arr)

    fen_arr, result_arr = get_positions(num_games, pgna, fen_arr, result_arr)

    #new_fen_arr, new_result_arr = get_positions(num_games, pgnb, fen_arr, result_arr)
    #fen_arr = np.concatenate((fen_arr, new_fen_arr), axis=None)
    #result_arr = np.concatenate((result_arr, new_result_arr), axis=None)
    new_fen_arr, new_result_arr = get_positions(num_games, pgnc, fen_arr, result_arr)
    fen_arr = np.concatenate((fen_arr, new_fen_arr), axis=None)
    result_arr = np.concatenate((result_arr, new_result_arr), axis=None)
    return fen_arr, result_arr

def fen_to_matrix(inputstr):
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    pieces = set(pieces_str)
    valid_spaces = set(range(1, 9))
    pieces_dict = {pieces_str[0]: 1, pieces_str[1]: 2, pieces_str[2]: 3, pieces_str[3]: 4,
                   pieces_str[4]: 5, pieces_str[5]: 6,
                   pieces_str[6]: -1, pieces_str[7]: -2, pieces_str[8]: -3, pieces_str[9]: -4,
                   pieces_str[10]: -5, pieces_str[11]: -6}

    boardtensor = np.zeros((8, 8, 6, 2))

    inputliste = inputstr.split()
    rownr = 0
    colnr = 0
    for i, c in enumerate(inputliste[0]):
        if c in pieces:
            if np.sign(pieces_dict[c]) == 1:
                boardtensor[rownr, colnr, np.abs(pieces_dict[c]) - 1, 0] = np.sign(pieces_dict[c])
            else:
                boardtensor[rownr, colnr, np.abs(pieces_dict[c]) - 1, 1] = -np.sign(pieces_dict[c])
            colnr = colnr + 1
        elif c == '/':  # new row
            rownr = rownr + 1
            colnr = 0
        elif int(c) in valid_spaces:
            colnr = colnr + int(c)
        else:
            raise ValueError("invalid fenstr at index: {} char: {}".format(i, c))
    return boardtensor

def get_board_matrix(fen):
    board = []
    for i in fen:
        board.append(fen_to_matrix(i))
    return board

def get_X_y(result, board):
    X = []
    y = []
    for i in range(len(board)):
        X.append(np.reshape(board[i], 768))
        y.append(result[i])
    return X, y

def logistic_reg(X_train, X_test, y_train, y_test, is_num_analysis):
    if is_num_analysis == 0:
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr = lr.fit(X_train, y_train)
        y_pred_test = lr.predict(X_test)
        y_targ_test = y_test
        y_pred_train = lr.predict(X_train)
        testing_acc = (y_pred_test == y_targ_test).sum() / len(y_pred_test)
        training_acc = (y_pred_train == y_train).sum() / len(y_pred_train)
        print(f'testing accuracy:  {testing_acc}')
        print(f'training accuracy:  {training_acc}')
        return lr
    else:
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr = lr.fit(X_train, y_train)
        y_pred_test = lr.predict(X_test)
        y_targ_test = y_test
        y_pred_train = lr.predict(X_train)
        testing_acc = (y_pred_test == y_targ_test).sum() / len(y_pred_test)
        training_acc = (y_pred_train == y_train).sum() / len(y_pred_train)
        training_error = np.mean(np.square(np.array(y_pred_train) - np.array(y_train)))
        testing_error = np.mean(np.square(np.array(y_pred_test) - np.array(y_targ_test)))
        return 1-training_acc, 1-testing_acc
    
def change_sparse_matrix_to_df(X, y):
    column_names = ["P", "N", "B", "R", "Q", "p", "n", "b", "r", "q", "win"]
    P_arr = []
    p_arr = []
    N_arr = []
    n_arr = []
    B_arr = []
    b_arr = []
    R_arr = []
    r_arr = []
    Q_arr =[]
    q_arr = []
    win = []
    #"PNBRQK"
    #boardtensor = np.zeros((8, 8, 6, 2))
    for matrix, value in zip(X, y):

        white = matrix[:, :, :, 0]
        black = matrix[:, :, :, 1]

        P = np.sum(np.reshape(white[:, :, 0], (64,)))
        p = np.sum(np.reshape(black[:, :, 0], (64,)))
        N = np.sum(np.reshape(white[:, :, 1], (64,)))
        n = np.sum(np.reshape(black[:, :, 1], (64,)))
        B = np.sum(np.reshape(white[:, :, 2], (64,)))
        b = np.sum(np.reshape(black[:, :, 2], (64,)))
        R = np.sum(np.reshape(white[:, :, 3], (64,)))
        r = np.sum(np.reshape(black[:, :, 3], (64,)))
        Q = np.sum(np.reshape(white[:, :, 4], (64,)))
        q = np.sum(np.reshape(black[:, :, 4], (64,)))
        P_arr.append(P)
        p_arr.append(p)
        N_arr.append(N)
        n_arr.append(n)
        B_arr.append(B)
        b_arr.append(b)
        R_arr.append(R)
        r_arr.append(r)
        Q_arr.append(Q)
        q_arr.append(q)
        if y == -1:
            y = 0
        win.append(value)
    pos = pd.DataFrame(columns=column_names)
    pos['P'] = P_arr
    pos['N'] = N_arr
    pos['B'] = B_arr
    pos['R'] = R_arr
    pos['Q'] = Q_arr
    pos['p'] = p_arr
    pos['n'] = n_arr
    pos['b'] = b_arr
    pos['r'] = r_arr
    pos['q'] = q_arr
    pos['win'] = win
    return pos

def chess_weights(pos):
    # define formula variables
    win = pos['win']
    P = pos['P']
    N = pos['N']
    B = pos['B']
    R = pos['R']
    Q = pos['Q']
    p = pos['p']
    n = pos['n']
    b = pos['b']
    r = pos['r']
    q = pos['q']

    # define formula
    formula = '''
        win ~ I(P - p) +
              I(N - n) +
              I(B - b) +
              I(R - r) +
              I(Q - q)
    '''

    # build model
    piece_weights = smf.glm(formula, family=sm.families.Binomial(), data=pos).fit()

    df = pd.read_html(piece_weights.summary().tables[1].as_html(), header=0, index_col=0)[0]
    print(df.head())
    pawn = df['coef'].values[1]
    knight = df['coef'].values[2]
    bishop = df['coef'].values[3]
    rook = df['coef'].values[4]
    queen = df['coef'].values[5]
    scale = 1/knight
    bishop = 3*scale*bishop
    pawn = 3*scale*pawn
    knight = 3*scale*knight
    rook = 3*scale*rook
    queen = 3*scale*queen
    print(f'pawn  = {pawn}')
    print(f'knight  = {knight}')
    print(f'bishop  = {bishop}')
    print(f'rook  = {rook}')
    print(f'queen  = {queen}')
    print(piece_weights.summary())

#1: 1,55,69
#1000, ()
def plot_two_graphs_one_fig(x, y1, y2):
    """
    Example function to draw multiple graphs on one matplotlib figure

    Some more examples with more rows:
    https://www.geeksforgeeks.org/plot-multiple-plots-in-matplotlib/
    """
    # just constructing some dummy data
    graph_one_ys = y1
    graph_two_ys = y2

    # alright. construct our Figure and Axes (refer to lab)
    # creating the space for 1 row and 2 columns of graphs

    # construct the dummy x values
    Xs = x
    assert len(Xs) == len(graph_one_ys) == len(graph_two_ys)

    # for the position 1, plot graph one
    plt.scatter(Xs, graph_one_ys, marker='o', label="train error", color="red")

    # set title at row 1 col 1
    plt.title("Classifier error filtered by number of moves")

    # set x and y labels
    plt.xlabel("num moves until the game ends")
    plt.ylabel("testing error")

    # for the position 2, plot graph 2
    plt.scatter(Xs, graph_two_ys, marker='x', label="test error", color="blue")
    plt.legend(loc="upper left")

    # set title at row 1 col 1
    # set x and y labels
    plt.grid()
    z = np.polyfit(Xs, graph_one_ys, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")
    m = np.polyfit(Xs, graph_two_ys, 1)
    p = np.poly1d(m)
    plt.plot(x, p(x), "b--")

    plt.show()


def num_moves_analysis(X, y):
    X_plt = []
    Y1 = []
    Y2 = []

    for move_num, index in move.items():
        #index = [1, 55, 66, 77, ...]
        x_new = []
        y_new = []
        for i in index:
            if len(index) > 500:
                x_new.append(X[i])
                y_new.append(y[i])

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x_new, y_new, test_size=0.30, random_state=0)
        testing_err, training_err = logistic_reg(X_train, X_test, y_train, y_test, 1)
        X_plt.append(move_num)
        Y1.append(testing_err)
        Y2.append(training_err)
    for i in range(len(X_plt)):
        print(f'move = {X_plt[i]}, test_err= {Y1[i]}, train_err = {Y2[i]}')

    plot_two_graphs_one_fig(X_plt, Y1, Y2)
        
            
def test_fide_finals(lr):
    board = get_board_matrix(k_vs_d)
    result = [1]*len(board)
    X, y = get_X_y(result, board)
    y_pred = lr.predict(X)
    acc = (y_pred).sum() / len(y_pred)
    print(f'the final game accuracy is : {acc}')

def bar_graph():
    # Numbers of pairs of bars you want
    N = 5

    # Data on X-axis

    # Specify the values of blue bars (height)
    blue_bar = (2.974624060150376, 3.0, 3.555451127819549, 5.340789473684211, 8.92951127819549)
    # Specify the values of orange bars (height)
    orange_bar = (1, 3, 3, 5, 9)

    # Position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10, 5))

    # Width of a bar
    width = 0.3

    # Plotting
    plt.bar(ind, blue_bar, width, label='Model Valuation')
    plt.bar(ind + width, orange_bar, width, label='Historical Capablanca Valuation')

    plt.xlabel('Chess Pieces')
    plt.ylabel('Values')
    plt.title('Learned Piece Values Compared to Historical Valuation')

    # xticks()
    # First argument - A list of positions at which ticks should be placed
    # Second argument -  A list of labels to place at the given locations
    plt.xticks(ind + width / 2, ('Pawn', 'Knight', 'Bishop', 'Rook', 'Queen'))

    # Finding the best position for legends and putting it
    plt.legend(loc='best')
    plt.show()

    plt.savefig('barplot.jpg')


def main():
    fen, result = get_fen(2000)

    board = get_board_matrix(fen)
    #b = game.board()
    #svg = chess.svg.board(board=b, size=400)
    X, y = get_X_y(result, board)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
    #lr = logistic_reg(X_train, X_test, y_train, y_test, 0)
    #test_fide_finals(lr)
    num_moves_analysis(X, y)
    #pos = change_sparse_matrix_to_df(board, y)
    #chess_weights(pos)
    #bar_graph()




if __name__=='__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

