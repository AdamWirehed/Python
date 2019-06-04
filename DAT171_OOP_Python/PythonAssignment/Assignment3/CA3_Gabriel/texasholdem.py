from Assignment3.CA3_Gabriel.cardlib3 import *
from Assignment3.CA3_Gabriel.gui import *
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSvg import *
from PyQt5.QtWidgets import *


class TexasHoldEmModel(QObject):
    """ A game model for the poker game of Texas Hold 'Em """
    new_table_data = pyqtSignal()
    new_player_data = pyqtSignal()
    new_status_data = pyqtSignal()
    game_ended = pyqtSignal()

    def __init__(self, player_names, buy_in):
        super().__init__()
        # Create objects of Players
        self.players = [Player(player_name, buy_in) for player_name in player_names]

        # Create table cards as hand
        self.table = HandModel()

        # Set start conditions
        self.round = 0
        self.active_player = -1

        # Start new round
        self.new_round()

    def new_round(self):
        """ If a new round is called upon, it is checked whether the players have money to keep playing,
        new cards are dealt and state-variables are reset """
        # Check game conditions
        if self.check_winner():
            self.game_ended.emit()  # Signal that the game is ended

        # Init new round
        self.pot = 0
        self.deck = StandardDeck()
        self.deck.shuffle()
        self.round += 1

        # Hand out cards
        for player in self.players:
            player.hand.cards = []
            player.hand.add_card(self.deck.take_card())
            player.hand.add_card(self.deck.take_card())

        # Empty the cards on the table
        self.table.cards = []

        # Set active player
        self.active_player = (self.round + 1)% 2

        # Define the last bet variable
        self.last_bet = -1

        # Counter to keep track of when to open turn and river card
        self.tracker = 0

        # Signal updates
        self.new_table_data.emit()  # Signal new data for the table
        self.new_player_data.emit()  # Signal new data for the players
        self.new_status_data.emit()  # Signal new data for the status bar

    @staticmethod
    def quit_game():
        """ Quits the application, only called upon after the winner is presented """
        sys.exit()

    def stop_game(self):
        """ Signaling that the game is stopped """
        self.game_ended.emit()

    def check_winner(self):
        """ Method for checking if all the players have enough money to keep playing """
        for player in self.players:
            if player.money == 0:
                return True

    def next_player(self):
        """
        Next players turn, setting the active player to the other
        """
        self.active_player = (self.active_player + 1) % 2  # Method to choose next player, and can go 'around'
        self.new_status_data.emit()

    def next_card(self):
        """
        Opens up the next card on the table, depending on how many there is. If 5 cards are opened, the method to
        compare the players card are called upon
        """
        # Reset last bet amount to -1 to indicate "new betting round"
        self.last_bet = -1

        # Update the tracker
        self.tracker += 1

        # Check whether to open turn och river card alt start new round
        if self.tracker == 1:
            self.flop()
        elif self.tracker == 2:
            self.turn()
        elif self.tracker == 3:
            self.river()
        else:
            self.evaluate()

    def flop(self):
        """
        Open up the flop cards
        """
        for card_count in range(3):
            self.table.add_card(self.deck.take_card())

        self.new_table_data.emit()  # TODO: Update table view

    def turn(self):
        """
        Open the turn card
        """
        self.table.add_card(self.deck.take_card())
        self.new_table_data.emit()  # TODO: Update table view

    def river(self):
        """
        Open the river card
        """
        self.table.add_card(self.deck.take_card())
        self.new_table_data.emit()  # TODO: Update table view

    def fold(self):
        """
        If the active player wants to fold
        """
        winning_player = (self.active_player + 1) % 2
        self.players[winning_player].add_money(self.pot)

        self.new_round()

    def raise_bet(self, amount):
        """
        When the active player wants to raise
        :param amount: $ to raise
        """
        if self.last_bet == 0:
            amount += self.last_bet
            cash_draw = -amount
            self.last_bet = amount

        elif self.last_bet == -1:
            cash_draw = -amount
            self.last_bet = amount

        else:
            amount += self.last_bet
            cash_draw = -amount
            self.last_bet = amount - self.last_bet

        self.pot += amount
        self.players[self.active_player].add_money(cash_draw)

        # Next players turn
        self.next_player()

        self.new_table_data.emit()
        self.new_player_data.emit()  # TODO: Update player view, table view

    def call(self):
        """
        When the active player wants to call the current bet
        """
        # If its the first action after opening new card
        if self.last_bet == -1:
            self.last_bet = 0

        # If last player checked
        elif self.last_bet == 0:
            self.next_card()

        # If last player did raise the bet
        else:

            self.raise_bet(0)
            self.next_card()

        self.next_player()

    def evaluate(self):
        """
        For the event when all cards are on the table and the hands are to be compared. The winner gets the pot,
        if the result is 'draw' the pot is splitted
        """
        player_hands = []
        # Retrieve each players best hand
        for player in self.players:
            player_hands.append(player.hand.best_poker_hand(self.table.cards))

        # Check what hand takes the win
        winning_hand = max(player_hands)

        # Check if other player has the same hand
        winning_players = []
        for player_index, player in enumerate(self.players):
            if player_hands[player_index] == winning_hand:
                winning_players.append(player_index)

        # Split the pot between the winners
        for player in winning_players:
            self.players[player].add_money(self.pot/len(winning_players))

        self.new_round()


class HandModel(Hand, QObject):
    """ A model for a set of cards, adjusted to be sent into CardView in the GUI """
    data_changed = pyqtSignal()

    def __init__(self):
        Hand.__init__(self)
        QObject.__init__(self)

        # Additional state needed by the UI, keeping track of the selected cards:
        self.marked_cards = [False]*len(self.cards)
        self.flipped_cards = True

    def flip(self):
        """ Flips the cards over, to hide them"""
        self.flipped_cards = not self.flipped_cards
        self.data_changed.emit()

    def flipped(self, i):
        """ This model only flips all or no cards, so we don't care about the index """
        # Might be different for other games though!
        return self.flipped_cards

    def add_card(self, card):
        """ Adds a card to the list in the model """
        super().add_card(card)
        self.data_changed.emit()  # Signal to update the cards


class Player(QObject):
    """ Represents a player, with a name, money and a set of cards """

    new_player_data = pyqtSignal()

    def __init__(self, player_name, buy_in):
        super().__init__()
        self.name = player_name
        self.money = buy_in
        self.hand = HandModel()

    def add_money(self, amount):
        """ Method for adding money to player """
        self.money += amount
        self.new_player_data.emit()  # Signal to update the player data


