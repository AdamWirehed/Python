class Game:
    def __init__(self, newPlayerName):
        self.playerName = newPlayerName
        self.playerHand = 'NA'
        self.botHand = 'NA'
        self.botHandStr = 'NA'
        self.winner = 'No winner yet'
        print('New instance of |Rock|Paper|Scissors| for ' + self.playerName)
        print('-----------------------------------')

    def chooseHand(self):
        self.playerHand = input('Choose your hand: ')
        import random
        self.botHand = random.randint(1, 4)

        if self.botHand == 1:
            self.botHandStr = 'Scissors'
        elif self.botHand == 2:
            self.botHandStr = 'Rock'
        elif self.botHand == 3:
            self.botHandStr = 'Paper'
        elif self.botHand == 4:
            self.botHandStr = 'Buddha'

    def runGame(self):
        if ((self.playerHand == 'rock' or self.playerHand == 'Rock') and self.botHand == 1):
            self.winner = 'You win'
        elif ((self.playerHand == 'rock' or self.playerHand == 'Rock') and self.botHand == 2):
            self.winner = 'You tie.'
        elif ((self.playerHand == 'rock' or self.playerHand == 'Rock') and self.botHand == 3):
            self.winner = 'You lose.'
        elif ((self.playerHand == 'paper' or self.playerHand == 'Paper') and self.botHand == 1):
            self.winner = 'You lose.'
        elif ((self.playerHand == 'paper' or self.playerHand == 'Paper') and self.botHand == 2):
            self.winner = 'You win!'
        elif ((self.playerHand == 'paper' or self.playerHand == 'Paper') and self.botHand == 3):
            self.winner = 'You tie.'
        elif ((self.playerHand == 'scissors' or self.playerHand == 'Scissors') and self.botHand == 1):
            self.winner = 'You tie.'
        elif ((self.playerHand == 'scissors' or self.playerHand == 'Scissors') and self.botHand == 2):
            self.winner = 'You lose.'
        elif ((self.playerHand == 'scissors' or self.playerHand == 'Scissors') and self.botHand == 3):
            self.winner = 'You win!'
        elif self.botHand == 4:
            self.winner = 'Buddha always wins'
        else:
            self.winner = 'INVALID input, try again'


def main(gamerIGN):
    myGame = Game(gamerIGN)
    myGame.chooseHand()

    print(myGame.playerName + ' choose ' + myGame.playerHand)
    print('Bot choose ' + str(myGame.botHandStr))

    myGame.runGame()

    print(myGame.winner)
    print()  # Empty line


hell = True  # Variable for looping game
playerName = input('State your IGN (In-Game-Name): ')
while hell:
    main(playerName)
    import time

    loop = input('Do you want to play again? ')
    if (loop == 'yes' or loop == 'Yes'):
        print('Rebooting..')
        print()
        time.sleep(2)
    else:
        hell = False  # Breaking the loop
