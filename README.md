# PokerHand_Classification_NN

Dataset link: https://archive.ics.uci.edu/ml/datasets/Poker+Hand

For this project I tried to determine the accuracy of a trained machine in predicting poker hands based on the input data of the two CSV dataset files: poker-hand-testing.data and poker-hand-training-true.data .

Attribute Information:

1) S1 "Suit of card #1"
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

2) C1 "Rank of card #1"
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

3) S2 "Suit of card #2"
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

4) C2 "Rank of card #2"
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

5) S3 "Suit of card #3"
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

6) C3 "Rank of card #3"
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

7) S4 "Suit of card #4"
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

8) C4 "Rank of card #4"
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

9) S5 "Suit of card #5"
Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}

10) C5 "Rank of card 5"
Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

11) CLASS "Poker Hand"
Ordinal (0-9)

0: Nothing in hand; not a recognized poker hand
1: One pair; one pair of equal ranks within five cards
2: Two pairs; two pairs of equal ranks within five cards
3: Three of a kind; three equal ranks within five cards
4: Straight; five cards, sequentially ranked with no gaps
5: Flush; five cards with the same suit
6: Full house; pair + different rank three of a kind
7: Four of a kind; four equal ranks within five cards
8: Straight flush; straight + flush
9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush

Training the datasets – MLPClassifier

	The first step in training the data is to split it and this is the most necessary step for this method.
In this manner, the X1 and X2 matrices will contain the inputs, meaning the suits and the ranks of the cards, and the T1 and T2 matrices will contain the output, meaning the player’s hand. I chose to initialize them X1, T1 and X2, T2 to avoid further errors and to assign correspondingly the data from each set, in this case 1 being an index for Testing dataset, while 2 representing the Training dataset
	The next step is to split the data further into xTrain, xTest, tTrain, tTest with the train_test_split function from the sklearn library. In our case, we will divide for both datasets:
•	xTrain1, xTest1, tTrain1, tTest1
•	xTrain2, xTest2, tTrain2, tTest2
After several runs, I found out that the size of the hidden layers of our neural network is decreasing the accuracy and we will try again many combinations to determine the best one.
For  the trainings  I used the following parameters from the MLPClassifier: 
•	alpha=1e-5 – regularization term parameter.
•	verbose=1 - whether to print progress messages to stdout.
•	Maximum iterations = 1000
•	hidden_layer_sizes = (15,15)
•	solver= ‘adam’ which is set by default.
•	random_state=1 - determines random number generation for weights and bias initialization, train-test split if early stopping is used, and batch sampling when solver='sgd' or 'adam'. Pass an int for reproducible results across multiple function calls
I chose the ‘adam’ solver because I read that it’s effective and achieves good results fast. 


As we increase the size of the hidden layers, we can see that the accuracy will increase and the confusion matrix will get better and better as the Neural Network start to predict more correctly the player's hands.
