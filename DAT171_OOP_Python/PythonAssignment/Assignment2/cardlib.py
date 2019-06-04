from random import shuffle
from enum import Enum, IntEnum
import abc
from collections import Counter


class Suit(IntEnum):
    """Assigns the different card suits a number"""
    Clubs = 0
    Hearts = 1
    Diamonds = 2
    Spades = 3


class CardValue(Enum):
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


class Hand:
    """Hand of cards. Functions to get, drop and sort cards"""
    def __init__(self):
        self.cards = []

    def __str__(self):  # Enables more elegant prints of hand objects
        text = ''

        for i, card in enumerate(self.cards):
            text += str(i) + ': ' + str(card) + '\n'
        return text

    # def get_cards(self, deck, noc):  # Input the deck to draw cards from and Number Of Cards (noc)
    #     '''
    #     Function that draw told number of cards from a specified deck
    #
    #     :type noc: list
    #     :type deck: object
    #     :param deck: The deck that cards will be drawn from
    #     :param noc: "Number Of Cards" the amount of cards that should be drawn
    #     '''
    #
    #     if len(deck.cards) > 0:
    #         self.cards.extend(deck.cards[0:noc])  # Extend the hand with number of cards that is drawn
    #         del deck.cards[0:noc]
    #     else:
    #         print("Deck is empty!")
    #         raise IndexError  # Raise error if the deck is empty when trying to draw a card

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
        return PokerHand(cards)


class StandardDeck:
    """Class for creating, shuffle and dealing cards from deck"""
    def __init__(self):
        self.cards = []

        for suit in Suit:                  # Creates deck with 52 cards where ace is value 1
            for value in range(2, 11):
                self.cards.append(NumberedCard(suit, value))
            self.cards.append(JackCard(suit))
            self.cards.append(QueenCard(suit))
            self.cards.append(KingCard(suit))
            self.cards.append(AceCard(suit))

    def shuffle(self):      # Imported function from random lib that shuffles the deck
        shuffle(self.cards)

    def deal_cards(self):
        if len(self.cards) > 0:
            return self.cards.pop(0)
        else:
            print("\nDeck is empty!")
            raise IndexError  # Raise error if the deck is empty when trying to draw a card

    def sort_cards(self):
        self.cards = sorted(self.cards, key=lambda card: card.get_value())


class PokerHandTypes(IntEnum):
    """Assigns the poker hands a value, lower = better"""
    Straight_flush = 9
    Four_of_a_kind = 8
    Full_house = 7
    Flush = 6
    Straight = 5
    Three_of_a_kind = 4
    Two_pair = 3
    One_pair = 2
    High_card = 1


class PokerHand:
    """Class just to collect all the check_poker_hand functions. All functions are static
    All functions can take more than 5 cards as input and yet find the best poker hand
    """

    def __init__(self, cards):
        """
        Check the hand after the best poker hand of all cards in the hand
        :return: best poker hand as object with highest card in the poker hand and highest card in the hand
        """
        self.poker_type = None
        self.card_value = None

        list_poker_hand = [PokerHand.check_straight_flush, PokerHand.check_four_of_a_kind,
                           PokerHand.check_full_house, PokerHand.check_flush,
                           PokerHand.check_straight, PokerHand.check_three_of_a_kind,
                           PokerHand.check_two_pair, PokerHand.check_pair, PokerHand.check_high_card]

        for poker_type, highest_card_value in zip(PokerHandTypes, list_poker_hand):
            if highest_card_value(cards):
                self.card_value = highest_card_value(cards)
                self.poker_type = poker_type
                break

    def __lt__(self, other):
        return [self.poker_type, self.card_value] < [other.poker_type, other.card_value]

    def __gt__(self, other):
        return [self.poker_type, self.card_value] > [other.poker_type, other.card_value]

    def __eq__(self, other):
        return [self.poker_type, self.card_value] == [other.poker_type, other.card_value]

    @staticmethod  # Two functions that will make counting combinations easier
    def value_count(cards):
        value_count = Counter([c.get_value() for c in cards])  # Get the number of one card value in cards
        return value_count

    @staticmethod
    def rank_count(cards):
        rank_count = Counter([c.get_suit() for c in cards])  # Get the number of one card suit in cards
        return rank_count

    @staticmethod
    def check_straight_flush(cards: list):
        """
        Checks for the best straight flush in a list of cards (may be more than just 5)

        :param cards: A list of playing cards.
        :return: None if no straight flush is found, else the PokerHand object and the value of the top card.
        """
        cards = sorted(cards, key=lambda card: card.get_value()) # Sort cards in order to give the method a chance
        vals = [(c.get_value(), c.suit) for c in cards] \
            + [(1, c.suit) for c in cards if c.get_value() == 14]  # Add the aces!
        for c in reversed(cards): # Starting point (high card)
            # Check if we have the value - k in the set of cards:
            found_straight = True
            for k in range(1, 5):
                if (c.get_value() - k, c.suit) not in vals:
                    found_straight = False
                    break
            if found_straight:
                return c.get_value()

    @staticmethod
    def check_full_house(cards: list):
        """
        Checks for the best full house in a list of cards (may be more than just 5)

        :param cards: A list of playing cards
        :return: None if no full house is found, else the PokerHand object and a tuple of the values of the
        triple and pair.
        """

        value_count = PokerHand.value_count(cards)

        # Find the card ranks that have at least three of a kind
        threes = [v[0] for v in value_count.items() if v[1] >= 3]
        threes.sort()
        # Find the card ranks that have at least a pair
        twos = [v[0] for v in value_count.items() if v[1] >= 2]
        twos.sort()

        # Threes are dominant in full house, lets check that value first:
        for three in reversed(threes):
            for two in reversed(twos):
                if two != three:
                    return three


    @staticmethod
    def check_four_of_a_kind(cards: list):
        """
        :param cards: A list of playing cards
        :return: PokerHand object and value of the four cards
        """
        value_count = PokerHand.value_count(cards)
        quad = [v[0] for v in value_count.items() if v[1] >= 4]
        if quad:
            return quad[0]

    @staticmethod
    def check_three_of_a_kind(cards: list):
        """
        :param cards: A list of playing cards
        :return: PokerHand object and value of the three cards
        """
        value_count = PokerHand.value_count(cards)
        threes = [v[0] for v in value_count.items() if v[1] >= 3]
        threes.sort()
        if threes:
            return threes[0]

    @staticmethod
    def check_pair(cards: list):
        """
        :return: PokerHand object and value of the two cards
        """
        value_count = PokerHand.value_count(cards)
        pair = [v[0] for v in value_count.items() if v[1] >= 2]
        pair.sort()
        if pair:
            return max(pair)

    @staticmethod
    def check_two_pair(cards: list):
        """Do two pairs
        :return: PokerHand object, value of the best pair and value of the second pair
        """
        # number_of_pairs = PokerHands.check_pair(self, cards)
        value_count = PokerHand.value_count(cards)
        pair = [v[0] for v in value_count.items() if v[1] >= 2]
        pair.sort()
        if len(pair) > 1:
            return pair[1]

    @staticmethod
    def check_flush(cards: list):
        """Do the flash function
        :return: PokerHand object and value of the highest card of the flush
        """
        rank_count = PokerHand.rank_count(cards)

        penta = [v[0] for v in rank_count.items() if v[1] >= 5]
        penta.sort()
        flush = []

        for c in cards:
            if penta and c.get_suit() == penta[0]:
                flush.append(c)

        flush.sort()

        if penta:
            return flush[-1].get_value()

    @staticmethod
    def check_straight(cards: list):
        """
        Checks for the best straight in a list of cards (may be more than just 5 cards)
        :param cards: A list of playing cards.
        :return: PokerHand object and the value of the top card.
        """
        cards = sorted(cards, key=lambda card: card.get_value())  # Sort cards in order to give the method a chance
        vals = [c.get_value() for c in cards] \
                + [1 for c in cards if c.get_value() == 14]  # Add the aces with value 1 as well
        for c in reversed(cards):  # Starting point (high card)
            # Check if we have the value - k in the set of cards:
            found_straight = True
            for k in range(1, 5):
                if (c.get_value() - k) not in vals:
                    found_straight = False
                    break
            if found_straight:
                highest = max(cards)
                return highest.get_value()

    @staticmethod
    def check_high_card(cards: list):
        '''
        :return: Poker hand object of value 8 and value of the highest card of cards
        '''
        cards = sorted(cards, key=lambda card: card.get_value())
        return cards[-1].get_value()
