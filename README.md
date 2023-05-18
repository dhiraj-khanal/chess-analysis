# chess-analysis
**Introduction**
This project aims to predict the outcome of chess games using machine learning techniques. It uses a logistic classifier to forecast the expected result (win or loss) based on a given chess position.

**Key Features**
Utilizes a sparse vector representation for the chess board state.
Employs logistic regression for game outcome prediction.
Benchmarked on a dataset containing 500,000 training and test samples.
Applies the concept of symmetry to handle different aspects of the game state like whose turn it is, castling rights, and en passant.
Derives piece value weights consistent with traditional chess theory.
Examines the effect of player errors or "blunders" on prediction accuracy.

**Methodology**
Chess positions are converted into Forsyth–Edwards Notation (FEN) strings, which are further transformed into a sparse matrix. Each entry in the sparse matrix corresponds to the presence or absence of a particular piece on a specific square. Logistic regression is then run on these vectors, generating weights that represent the estimated value of each chess piece.

**Results**
The system was trained on a large dataset and achieved approximately 68% accuracy on the training set and 69% on the test set. The learned piece weights closely matched the traditional chess piece values, which gives credence to the model's predictions. Furthermore, a significant portion of the prediction error was found to be due to human errors in the game data.
