from enum import Enum
import abc
from random import shuffle


class Suit(Enum):
    """The 4 different suits"""
    Clubs, Diamonds, Hearts, Spades = 0, 1, 2, 3


class PokerHandTypes(Enum):
    """The different types of poker hands"""
    straight_flush = 9
    four_of_a_kind = 8
    full_house = 7
    flush = 6
    straight = 5
    three_of_a_kind = 4
    pair = 3
    par = 2
    high_card = 1


class PlayingCard(metaclass=abc.ABCMeta):
    """A standard playing card with a rank, value and suit. E.g; rank: King, value; 13, suit: Hearts"""
    def __init__(self, suit):
        self.suit = suit

    def get_suit(self):
        """ Returns the suit of the card"""
        return self.suit

    @abc.abstractmethod
    def get_rank(self):
        """Returns the rank of the card"""

    @abc.abstractmethod
    def get_value(self):
        """Returns the value of the rank"""

    def __lt__(self, other):
        return (self.get_value(), self.get_suit().value) < (other.get_value(), other.get_suit().value)

    def __eq__(self, other):
        return (self.get_value(), self.get_suit().value) == (other.get_value(), other.get_suit().value)

    def __ge__(self, other):
        return (self.get_value(), self.get_suit().value) >= (other.get_value(), other.get_suit().value)

    def __str__(self):
        return '{} of {}'.format(self.get_rank(), self.suit.name)


class NumberedCard(PlayingCard):
    """Cards with value 2-10 and suit"""
    def __init__(self, rank, suit):
        super().__init__(suit)
        self.rank = rank

    def get_rank(self):
        return self.rank

    def get_value(self):
        return self.rank


class JackCard(PlayingCard):
    """Jack card with value 11 and suit"""

    def get_rank(self):
        return 'Jack'

    def get_value(self):
        return 11


class QueenCard(PlayingCard):
    """Queen card with  value 12 and suit"""

    def get_rank(self):
        return 'Queen'

    def get_value(self):
        return 12


class KingCard(PlayingCard):
    """King card with value 13 and suit"""

    def get_rank(self):
        return 'King'

    def get_value(self):
        return 13


class AceCard(PlayingCard):
    """Ace card with value 14 and suit"""

    def get_rank(self):
        return 'Ace'

    def get_value(self):
        return 14


class StandardDeck:
    """ A standard 52 card deck """
    class EmptyDeckError(Exception):
        pass

    def __init__(self):
        possible_values = range(2, 11)
        self.cards = []
        for suit in Suit:
            for value in possible_values:
                self.cards.append(NumberedCard(value, suit))
            self.cards.append(JackCard(suit))
            self.cards.append(QueenCard(suit))
            self.cards.append(KingCard(suit))
            self.cards.append(AceCard(suit))

    def shuffle(self):
        """
        Shuffles the cards in the deck.
        :return:
        """
        shuffle(self.cards)

    def take_card(self):
        """
        Removes the top card of the deck.
        :return: Playing card removed from deck.
        """
        if len(self.cards) == 0:
            raise StandardDeck.EmptyDeckError
        card_taken = self.cards[0]
        self.cards = self.cards[1:]

        return card_taken

    def __str__(self):
        lo_cards = '<Number of Cards: {}>\n'.format(len(self.cards))
        for card in self.cards:
            lo_cards += str(card) + '\n'
        return lo_cards


class Hand:
    """Describes what cards a player has on hand"""

    class NoCardsInHandError(Exception):
        pass

    def __init__(self, cards=None):
        if cards is None:
            self.cards = []
        else:
            self.cards = cards

    def add_card(self, card):
        """
        Adds a card to the hand.
        :param card: A playing card
        :return:
        """
        self.cards.append(card)

    def drop_cards(self, card_index):
        """
        Removes card(s) from the hand, specified by index.
        :param card_index: The index of the card(s) to drop.
        :return:
        """
        if type(card_index) == list:
            cards_td = []
            for index in card_index:
                cards_td.append(self.cards[index])

            for card in cards_td:
                self.cards.remove(card)
        else:
            self.cards.remove(self.cards[card_index])

    def sort_cards(self):
        """
        Sorts the playing cards in the hand in descending order.
        :return:
        """
        self.cards = sorted(self.cards, reverse=True)

    def best_poker_hand(self, cards_on_table=None):
        """
        Finds the best poker hand in the hand.
        :param cards_on_table: List of PlayingCards on the table
        :return: A PokerHand describing the best poker hand.
        """

        from poker_hand_check_functions import (check_full_house, check_straight_flush, check_four_of_a_kind,
                                                check_flush, check_straight, check_three_of_a_kind, check_pair,
                                                check_par, check_high_card)

        # Check if hand has got cards
        if len(self.cards) == 0:
            raise Hand.NoCardsInHandError

        # Create a list of cards, combining cards on the hand with cards on the table
        cards_to_check = self.cards
        if cards_on_table:
            cards_to_check.extend(cards_on_table)

        hand_type = PokerHandTypes.straight_flush
        high = check_straight_flush(cards_to_check)
        if high is None:
            hand_type = PokerHandTypes.four_of_a_kind
            high = check_four_of_a_kind(cards_to_check)
            if high is None:
                hand_type = PokerHandTypes.full_house
                high = check_full_house(cards_to_check)
                if high is None:
                    hand_type = PokerHandTypes.flush
                    high = check_flush(cards_to_check)
                    if high is None:
                        hand_type = PokerHandTypes.straight
                        high = check_straight(cards_to_check)
                        if high is None:
                            hand_type = PokerHandTypes.three_of_a_kind
                            high = check_three_of_a_kind(cards_to_check)
                            if high is None:
                                hand_type = PokerHandTypes.pair
                                high = check_pair(cards_to_check)
                                if high is None:
                                    hand_type = PokerHandTypes.par
                                    high = check_par(cards_to_check)
                                    if high is None:
                                        hand_type = PokerHandTypes.high_card
                                        high = check_high_card(cards_to_check)

        return PokerHand(hand_type, high)

    def __str__(self):
        lo_cards = ''
        for i, card in enumerate(self.cards):
            lo_cards += str(i) + ': ' + str(card) + '\n'
        return lo_cards


class PokerHand:
    """Describes what kind of poker hand, e.g: straight, three-of-a-kind, and what the highest card is"""

    def __init__(self, poker_hand_type, highest_card):
        self.poker_hand_type = poker_hand_type
        self.highest_card = highest_card

    def __lt__(self, other):
        return (self.poker_hand_type.value, self.highest_card) < (other.poker_hand_type.value, other.highest_card)

    def __eq__(self, other):
        return (self.poker_hand_type.value, self.highest_card) == (other.poker_hand_type.value, other.highest_card)

    def __ge__(self, other):
        return (self.poker_hand_type.value, self.highest_card) >= (other.poker_hand_type.value, other.highest_card)

    def __str__(self):

            return str(self.poker_hand_type.name) + ' of/with/to ' + str(self.highest_card) + '(s)'
