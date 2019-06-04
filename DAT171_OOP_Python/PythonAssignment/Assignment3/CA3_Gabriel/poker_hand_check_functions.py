from collections import Counter  # Counter is convenient for counting objects (a specialized dictionary)


def check_straight_flush(cards):
    """
    Checks for the best straight flush in a list of cards (may be more than just 5)

    :param cards: A list of playing cards.
    :return: None if no straight flush is found, else the value of the top card.
    """
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


def check_four_of_a_kind(cards):
    """
    Checks for the best four of a kind in a list of cards (may be more than just 5)

    :param cards: A list of playing cards
    :return: None if no four of a kind is found, else a list of the values of which four of a kind is found.
    """
    value_count = Counter()
    for card in cards:
        value_count[card.get_value()] += 1
    fours = [v[0] for v in value_count.items() if v[1] >= 4]

    if fours:
        fours.reverse()
        fours = fours[0]
        values = [v.get_value() for v in cards if v.get_value() != fours]
        try:
            return fours, max(values)
        except ValueError:
            return fours


def check_full_house(cards):
    """
    Checks for the best full house in a list of cards (may be more than just 5)

    :param cards: A list of playing cards
    :return: None if no full house is found, else a tuple of the values of the triple and pair.
    """
    value_count = Counter()
    for c in cards:
        value_count[c.get_value()] += 1
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
                return three, two


def check_flush(cards):
    """
    Checks for the best flush in a list of cards

    :param cards: A list of playing cards
    :return: None if no flush is found, else a list of the suits for which flush is found in descending order.
    """
    value_count = Counter()
    for card in cards:
        value_count[card.suit] += 1
    flush = [v[0].name for v in value_count.items() if v[1] >= 5]

    if flush:
        high_card = max([card.get_value() for card in cards if card.get_suit().name == flush[0]])
        return high_card


def check_straight(cards):
    """
       Checks for the best straight in a list of cards (may be more than just 5)

       :param cards: A list of playing cards.
       :return: None if no straight is found, else the value of the top card.
       """
    vals = [(c.get_value()) for c in cards] \
           + [1 for c in cards if c.get_value() == 14]  # Add the aces!
    for c in reversed(cards):  # Starting point (high card)
        # Check if we have the value - k in the set of cards:
        found_straight = True
        for k in range(1, 5):
            if (c.get_value() - k) not in vals:
                found_straight = False
                break
        if found_straight:
            return c.get_value()


def check_three_of_a_kind(cards):
    """
    Checks for the best three of a kind in a list of cards (may be more than just 5)

    :param cards: A list of playing cards
    :return: None if no three of a kind is found, else a list of the values of which three of a kind is found.
    """
    value_count = Counter()
    for card in cards:
        value_count[card.get_value()] += 1
    threes = [v[0] for v in value_count.items() if v[1] >= 3]
    if threes:
        threes.reverse()
        best_threes = threes[0]

        values = [v.get_value() for v in cards if v.get_value() != best_threes]

        try:
            return best_threes, max(values)
        except ValueError:
            return best_threes


def check_pair(cards):
    """
    Checks for the best pair in a list of cards (may be more than just 5)

    :param cards: A list of playing cards
    :return: None if no pair is found, else a tuple of the values of the pair and highest card.
    """

    value_count = Counter()
    for c in cards:
        value_count[c.get_value()] += 1
    # Find the card ranks that have at least a pair
    secondtwos = [v[0] for v in value_count.items() if v[1] >= 2]
    secondtwos.sort()
    # Find the other card ranks that have at least a pair
    twos = [v[0] for v in value_count.items() if v[1] >= 2]
    twos.sort()

    for secondtwo in reversed(secondtwos):
        for two in reversed(twos):
            if two != secondtwo:
                values = [v.get_value() for v in cards if (v.get_value() != two and v.get_value() != secondtwo)]

                try:
                    return secondtwo, two, max(values)
                except ValueError:
                    return secondtwo, two


def check_par(cards):
    """
    Checks for the best par in a list of cards (may be more than just 5)

    :param cards: A list of playing cards
    :return: None if no par is found, else a list of the values of which a par is found.
    """
    value_count = Counter()
    # Saves the counts of each values
    for card in cards:
        value_count[card.get_value()] += 1
    twos = [v[0] for v in value_count.items() if v[1] >= 2]
    twos.reverse()
    if twos:
        twos = sorted(twos, reverse=True)
        best_two = twos[0]

        values = [v.get_value() for v in cards if v.get_value() != best_two]

        try:
            return best_two, max(values)
        except ValueError:
            return best_two


def check_high_card(cards):
    """
    Checks for the highest card in a list of cards (may be more than just 5).

    :param cards: A list of playing cards.
    :return: Sorted tuple with the values of the card.
    """
    cards = sorted(cards, reverse=True)
    return cards[0]
