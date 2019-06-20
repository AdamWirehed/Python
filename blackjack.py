# Script of the game Blackjack
# Using classes to decide the rules, number och decks in play and the course of
# the game

import numpy as np


class Blackjack:
    """Stating the rules, number of decks in play and the course of the game"""

    def __init__(self, newPlayername):
        self.playerName = newPlayername
        self.playerHand = 'NA'
        self.playerHandstr = 'NA'
        self.playerChoice = 'NA'
        self.houseHand = 'NA'
        self.houseHandstr = 'NA'
        self.winner = 'No winner yet'
        print('New instance of Blackjack for ' + self.playerName)
        print('-------------------------------------------------')

    def setupDeck(self):
        """Choosing the amount of decks in play"""
        quarter = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        deck = np.array([quarter, quarter, quarter, quarter])  # Creating deck

        nrDecks = input('How many decks should be in play? (1-4): ')
        print()

        self.dInplay = np.repeat(deck, nrDecks)
        # Stating how many decks will be in play

    def dealingCards(self):
        """Card dealing"""
        import random
        posCard1 = random.randint(0, len(self.dInplay))
        card1 = self.dInplay[posCard1]
        np.delete(self.dInplay, posCard1)
        posCard2 = random.randint(0, len(self.dInplay))
        card2 = self.dInplay[posCard2]
        np.delete(self.dInplay, posCard2)
        # Dealing the cards to the player and deleting them from the deck

        import time
        print('Dealing cards...')
        print()
        time.sleep(2)
        self.playerHand = card1 + card2
        self.playerHandstr = 'Player hand: ' + str(card1) + ' and ' \
            + str(card2)
        print(self.playerHandstr)  # Showing the player his cards

        self.session = 'Start'
        while self.playerHand < 21 and self.session != 'Over':
            self.playerChoice = input('Hit or stand?: ')
            # Letting the player to chose to hit or stand

            if self.playerChoice == 'hit':
                posCardNew = random.randint(0, len(self.dInplay))
                cardNew = self.dInplay[posCardNew]
                np.delete(self.dInplay, posCardNew)
                # Dealing the player a new card and removing it from the deck

                self.playerHandstr = 'Player hand: ' + str(self.playerHand) + \
                    ' and ' + str(cardNew)
                self.playerHand = self.playerHand + cardNew
                print(self.playerHandstr)  # Showing the player the card
                print()

            elif self.playerChoice == 'stand':
                posHouse1 = random.randint(0, len(self.dInplay))
                house1 = self.dInplay[posHouse1]
                np.delete(self.dInplay, posHouse1)
                posHouse2 = random.randint(0, len(self.dInplay))
                house2 = self.dInplay[posHouse2]
                np.delete(self.dInplay, posHouse2)
                # Dealing cards to the house and deleting them from the deck

                self.houseHand = house1 + house2
                self.houseHandstr = 'House hand: ' + str(house1) + ' and ' \
                    + str(house2)
                print(self.houseHandstr)  # Showing the house's cards
                time.sleep(2)

                while self.houseHand < self.playerHand:
                    posHouseNew = random.randint(0, len(self.dInplay))
                    houseNew = self.dInplay[posHouseNew]
                    np.delete(self.dInplay, posHouseNew)
                    self.houseHandstr = 'House hand: ' + str(self.houseHand) +\
                        ' and ' + str(houseNew)
                    self.houseHand = self.houseHand + houseNew
                    print('Dealing card...')
                    print(self.houseHandstr)
                    time.sleep(2)

                if (self.houseHand > self.playerHand and self.houseHand <= 21):
                    print('House closer to 21. You lose.')
                    self.session = 'Over'
                    print()

                elif self.houseHand > 21:
                    print('House over 21. You win!')
                    self.session = 'Over'
                    print()

        if self.playerHand == 21:
            print('Blackjack! You win!')

        elif self.playerHand > 21:
            print('Player hand value over 21. You lose.')


playerName = input('State your IGN (In-Game-Name): ')


def main(gamerIGN):
    """Running the game"""
    myGame = Blackjack(gamerIGN)
    myGame.setupDeck()

    hell = True
    while hell:
        myGame.dealingCards()

        loop = input('Want to play again?: ')
        if loop == 'yes':
            hell == True
        elif loop == 'no':
            hell = False


main(playerName)
