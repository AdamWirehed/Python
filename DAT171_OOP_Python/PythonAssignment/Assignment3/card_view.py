from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSvg import *
from PyQt5.QtWidgets import *

from Assignment2 import cardlib
import enum
import abc

# NOTE: This is just given as an example of how to use CardView.
# It is expected that you will need to adjust things to make a game out of it. 
# Some things can be removed, other things modified.


class TableScene(QGraphicsScene):
    """ A scene with a table cloth background """
    def __init__(self):
        super().__init__()
        self.tile = QPixmap('cards/table.png')
        self.setBackgroundBrush(QBrush(self.tile))


class CardItem(QGraphicsSvgItem):
    """ A simple overloaded QGraphicsSvgItem that also stores the card position """
    def __init__(self, renderer, position):
        super().__init__()
        self.setSharedRenderer(renderer)
        self.position = position


class CardView(QGraphicsView):
    """ A View widget that represents the table area displaying a players cards. """

    # Underscores indicate a private function/method!
    def __read_cards(): # Ignore the PyCharm warning on this line. It's correct.
        """
        Reads all the 52 cards from files.
        :return: Dictionary of SVG renderers
        """
        all_cards = dict() # Dictionaries let us have convenient mappings between cards and their images
        for suit_file, suit in zip('HDSC', range(4)): # Check the order of the suits here!!!
            for value_file, value in zip(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'], range(2, 15)):
                file = value_file + suit_file
                key = (value, suit)  # I'm choosing this tuple to be the key for this dictionary
                all_cards[key] = QSvgRenderer('cards/' + file + '.svg')
        return all_cards

    # We read all the card graphics as static class variables
    back_card = QSvgRenderer('cards/Red_Back_2.svg')
    all_cards = __read_cards()

    def __init__(self, cards_model, card_spacing=250, padding=10):
        """
        Initializes the view to display the content of the given model
        :param cards_model: A model that represents a set of cards.
        The model should have: data_changed, cards, clicked_position, flipped,
        :param card_spacing: Spacing between the visualized cards.
        :param padding: Padding of table area around the visualized cards.
        """
        self.scene = TableScene()
        super().__init__(self.scene)

        self.model = cards_model
        self.card_spacing = card_spacing
        self.padding = padding

        # Whenever the this window should update, it should call the "change_cards" method.
        # This can, for example, be done by connecting it to a signal.
        # The view can listen to changes:
        cards_model.new_cards.connect(self.change_cards)
        # It is completely optional if you want to do it this way, or have some overreaching Player/GameState
        # call the "change_cards" method instead. z

        # Add the cards the first time around to represent the initial state.
        self.change_cards()

    def change_cards(self):
        # Add the cards from scratch
        self.scene.clear()
        for i, card in enumerate(self.model.cards):
            # The ID of the card in the dictionary of images is a tuple with (value, suit), both integers
            # TODO: YOU MUST CORRECT THE EXPRESSION TO MATCH YOUR PLAYING CARDS!!!
            # TODO: See the __read_cards method for what mapping are used.
            graphics_key = (card.get_value(), card.get_suit().value)
            renderer = self.back_card if self.model.flipped(i) else self.all_cards[graphics_key]
            c = CardItem(renderer, i)

            # Shadow effects are cool!
            shadow = QGraphicsDropShadowEffect(c)
            shadow.setBlurRadius(10.)
            shadow.setOffset(5, 5)
            shadow.setColor(QColor(0, 0, 0, 180)) # Semi-transparent black!
            c.setGraphicsEffect(shadow)

            # Place the cards on the default positions
            c.setPos(c.position * self.card_spacing, 0)
            # Sets the opacity of cards if they are marked.
            self.scene.addItem(c)

        self.update_view()

    def update_view(self):
        scale = (self.viewport().height()-2*self.padding)/313
        self.resetTransform()
        self.scale(scale, scale)
        # Put the scene bounding box
        self.setSceneRect(-self.padding//scale, -self.padding//scale,
                          self.viewport().width()//scale, self.viewport().height()//scale)

    def resizeEvent(self, painter):
        # This method is called when the window is resized.
        # If the widget is resize, we gotta adjust the card sizes.
        # QGraphicsView automatically re-paints everything when we modify the scene.
        self.update_view()
        super().resizeEvent(painter)

    # You can remove these events if you don't need them.
    def mouseDoubleClickEvent(self, event):
        self.model.flip() # Another possible event. Lets add it to the flip functionality for fun!


# We use the cards defined in our own library here:
class Suit(enum.IntEnum):
    """Assigns the different card suits a number"""
    Clubs = 1
    Hearts = 2
    Diamonds = 3
    Spades = 4


class CardValue(enum.Enum):
    """Assigns the face cards a number"""
    Jack = 11
    Queen = 12
    King = 13
    Ace = 14


class PlayingCard(metaclass=abc.ABCMeta):
    """Superclass PlayingCard, creates init for suit and two abstract functions"""
    def __init__(self, suit):
        self.suit = Suit(suit)
        self.suit_symbols = {'Spades': '♠', 'Diamonds': '♦', 'Hearts': '♥', 'Clubs': '♣'}

    @abc.abstractmethod
    def get_value(self):
        """Returns the value of card"""
        return self.value

    def get_suit(self):
        """Returns the value of the rank"""
        return self.suit

    def __lt__(self, other):
        """Implements lesser than operator"""
        return self.get_value() < other.get_value()

    def __gt__(self, other):
        """Implements greater than operator"""
        return self.get_value() > other.get_value()

    def __eq__(self, other):
        """Implements equal to operator"""
        return self.get_value() == other.get_value()


class NumberedCard(PlayingCard):
    """Subclass to PlayingCard, expands init to values as well. Functions get_value and get_suit to
        get integers representations for values and suits"""
    def __init__(self, suit, value):
        PlayingCard.__init__(self, suit)
        self.value = value

    def __str__(self):
        return '{} of {}'.format(self.value, self.suit_symbols[str(Suit(self.suit).name)])

    def get_value(self):
        return self.value


class JackCard(PlayingCard):

    def __str__(self):
        return '{} of {}' .format(CardValue(self.get_value()).name, self.suit_symbols[str(Suit(self.suit).name)])

    def get_value(self):
        return 11


class QueenCard(PlayingCard):

    def __str__(self):
        return '{} of {}' .format(CardValue(self.get_value()).name, self.suit_symbols[str(Suit(self.suit).name)])

    def get_value(self):
        return 12


class KingCard(PlayingCard):

    def __str__(self):
        return '{} of {}' .format(CardValue(self.get_value()).name, self.suit_symbols[str(Suit(self.suit).name)])

    def get_value(self):
        return 13


class AceCard(PlayingCard):

    def __str__(self):
        return '{} of {}' .format(CardValue(self.get_value()).name, self.suit_symbols[str(Suit(self.suit).name)])

    def get_value(self):
        return 14


# Our Hand class from assignment 2
class Hand:
    """Hand of cards. Functions to get, drop and sort cards"""
    def __init__(self):
        self.cards = []
        # deck = cardlib.StandardDeck()
        # self.get_cards(deck, 2)

    def __str__(self):  # Enables more elegant prints of hand objects
        text = ''

        for i, card in enumerate(self.cards):
            text += str(i) + ': ' + str(card) + '\n'
        return text

    def get_cards(self, deck, noc):  # Input the deck to draw cards from and Number Of Cards (noc)
        '''
        Function that draw told number of cards from a specified deck

        :type noc: list
        :type deck: object
        :param deck: The deck that cards will be drawn from
        :param noc: "Number Of Cards" the amount of cards that should be drawn
        '''

        if len(deck.cards) > 0:
            self.cards.extend(deck.cards[0:noc])  # Extend the hand with number of cards that is drawn
            del deck.cards[0:noc]
        else:
            print("Deck is empty!")
            raise IndexError  # Raise error if the deck is empty when trying to draw a card

    def add_card(self, card):
        self.cards.append(card)

    def drop_cards(self, index_cards):
        '''
        Function that takes one positional index as input and delete the card on that position in hand
        :type index_cards: list
        :param index_cards: The position of the cards that will be removed
        '''

        try:
            for index in sorted(index_cards, reverse=True):
                del self.cards[index]
        except IndexError:
            print("No card on that position!")


    def sort_cards(self):
        '''
        Function that sort cards either by value or suit, the player decides which one
        '''
        self.cards = sorted(self.cards, key=lambda card: card.get_value())

        sort = input('Sort hand by value or suit? ')

        if sort == ('value' or 'Value'):
            self.cards = sorted(self.cards, key=lambda card: card.get_value())

        elif sort == ('suit' or 'Suit'):
            self.cards = sorted(self.cards, key=lambda card: card.get_suit())

        else:
            print('Input error')

    def best_poker_hand(self, cards):
        """
        Check the hand after the best poker hand of all cards in the hand
        :return: best poker hand as object with highest card in the poker hand and highest card in the hand
        """
        cards.extend(self.cards)
        return cardlib.PokerHand(cards)


# We can extend this class to create a model, which updates the view whenever it has changed.
# NOTE: You do NOT have to do it this way.
# You might find it easier to make a Player-model, or a whole GameState-model instead.
# This is just to make a small demo that you can use. You are free to modify
class HandModel(Hand, QObject):
    new_cards = pyqtSignal()

    def __init__(self):
        Hand.__init__(self)
        QObject.__init__(self)

        # Additional state needed by the UI, keeping track of the selected cards:
        self.marked_cards = [False]*len(self.cards)
        self.flipped_cards = True

    def flip(self):
        # Flips over the cards (to hide them)
        self.flipped_cards = not self.flipped_cards
        self.new_cards.emit()

    def flipped(self, i):
        # This model only flips all or no cards, so we don't care about the index.
        # Might be different for other games though!
        return self.flipped_cards

    def add_card(self, card):
        super().add_card(card)
        self.new_cards.emit()

    def clear(self):
        self.cards.clear()
        self.new_cards.emit()