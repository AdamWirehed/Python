from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtSvg import *
import sys
from DAT171_Python_assignments.CA3 import card_view
from DAT171_Python_assignments.CA2 import cardlib


class Player(QObject):
    """ A player object with name and amount of credits. It can win, bet or clear the credits. """
    new_credits = pyqtSignal()

    def __init__(self, name, buyIn):
        super().__init__()
        self.name = name
        self.credits = buyIn
        self.total_bet = 0
        self.hand = card_view.HandModel()

    def win(self, amount):
        self.credits += amount.credits
        self.total_bet = 0
        self.new_credits.emit()

    def bet(self, amount):
        self.credits -= amount
        self.total_bet += amount
        self.new_credits.emit()

    def clear(self):
        self.total_bet = 0
        self.new_credits.emit()


class PotModel(QObject):
    """ A model for the pot. In order to clear and add to the pot. """
    new_value = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.credits = 0

    def clear(self):
        self.credits = 0
        self.new_value.emit()

    def __iadd__(self, amount):
        self.credits += amount
        self.new_value.emit()


class TableModel(QObject):
    """ A model for the poker table holding up to five cards. It can clear the table, flip the cards or add cards. """
    new_cards = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.cards = []

    def clear(self):
        self.cards.clear()
        self.new_cards.emit()

    def add_cards(self, cards):
        self.cards += cards
        self.new_cards.emit()

    def flip(self):
        pass

    def flipped(self, i):
        pass


class TexasHoldemMode(QObject):
    """ The game model with specific logic for the poker game Texas Hold Em. """

    # Define the signals
    new_current_player = pyqtSignal()
    game_ended = pyqtSignal()
    player_win = pyqtSignal()

    def __init__(self, playernames: list, BuyIn: int):
        super().__init__()
        # Define all the variables that will be used
        self.players = [Player(name_player, BuyIn) for name_player in playernames] # List of player objects
        self.deck = None
        self.table_cards = TableModel()
        self.round = 0
        self.active_player = 0
        self.pot = PotModel()
        self.last_bet = 0
        self.last_call = False
        self.winning_hand = None
        self.winning_player_index = 0

        # Start a new poker round
        self.new_round()

    def new_round(self):
        """ Start a new poker round. """
        for player in self.players:
            if player.credits == 0:
                self.exit_game()

        # Define parameters for a new round
        self.round += 1
        self.last_bet = 0
        self.pot.clear()
        self.last_call = False

        self.deck = cardlib.StandardDeck()
        self.deck.shuffle()

        # Deal the cards
        for player in self.players:
            player.hand.cards.clear()
            player.hand.get_cards(self.deck, 2)

        self.table_cards.clear()

    def get_active_player(self) -> Player:
        """ Returns the active player object """
        return self.players[self.active_player]

    def bet(self, amount):
        """ Method for placing bets. """
        if self.last_bet == 0:
            self.pot.credits += amount
            self.get_active_player().bet(amount)
            self.last_bet = amount
            self.last_call = False

        elif self.last_bet < amount:
            self.pot.credits += amount
            self.get_active_player().bet(amount)
            self.last_bet = amount
            self.last_call = False

        elif self.last_bet == amount:
            self.pot.credits += amount
            self.get_active_player().bet(amount)
            self.last_bet = amount
            self.next_card()

        elif self.get_active_player().total_bet + amount == self.pot.credits:
            self.pot.credits += amount
            self.get_active_player().bet(amount)
            self.last_bet = amount
            self.next_card()

        else:
            raise ValueError('Last bet higher than this bet!')

        self.next_player()
        self.pot.new_value.emit()

    def call(self):
        """ Method for calling a round. """
        if self.last_bet == 0 and not self.last_call:
            self.last_call = True
            self.next_player()

        elif self.last_bet != 0 and not self.last_call:
            self.last_call = True
            self.bet(self.last_bet)
        else:
            self.next_card()
            self.next_player()

    def fold(self):
        winning_player = (self.active_player + 1) % 2
        self.players[winning_player].win(self.pot)
        self.new_round()
        self.next_player()

    def all_in(self):
        """ Method for betting the remaining credits """
        self.bet(self.get_active_player().credits)
        self.last_bet = self.get_active_player().credits
        self.last_call = False
        self.pot.new_value.emit()

    def next_card(self):
        if len(self.table_cards.cards) == 5:
            self.evaluate() # If five cards are up it is time to evaluate the winner
            self.new_round()

        elif len(self.table_cards.cards) == 0: # Three cards up the first time
            self.table_cards.add_cards([self.deck.deal_cards(),
                                        self.deck.deal_cards(),
                                        self.deck.deal_cards()])
        else:
            self.table_cards.add_cards([self.deck.deal_cards()])

        self.last_call = False
        self.last_bet = 0

    def next_player(self):
        self.active_player = (1 + self.active_player) % 2
        self.new_current_player.emit()

    def evaluate(self):
        """ Method for evaluating wich player that wins. """
        index_player = 0
        winning_hand = None

        for i, player in enumerate(self.players):
            current_pokerhand = player.hand.best_poker_hand(self.table_cards.cards)
            if i == 0:
                winning_hand = current_pokerhand
            elif winning_hand < current_pokerhand:
                winning_hand = current_pokerhand
                index_player = i
            else:
                pass
        self.winning_hand = winning_hand
        self.winning_player_index = index_player
        self.players[index_player].win(self.pot)
        self.player_win.emit()

    @staticmethod
    def exit_game():
        sys.exit()

    def stop_game(self): # Method to stop the game, for future implementations
        self.game_ended.emit()


