import Assignment2.cardlib as cl


def test_operators():
    deckOfCards = cl.StandardDeck()
    hand = cl.Hand()
    hand.get_cards(deckOfCards, 12)
    print(hand)
    less_true = hand.cards[0] < hand.cards[1]
    assert less_true == True
    less_false = hand.cards[3] < hand.cards[0]
    assert less_false == False

    eq_true = hand.cards[0] == hand.cards[0]
    assert eq_true == True
    eq_false = hand.cards[1] == hand.cards[2]
    assert eq_false == False

    gt_true = hand.cards[4] > hand.cards[3]
    assert gt_true == True
    gt_false = hand.cards[4] > hand.cards[5]
    assert gt_false == False


def test_get_card():
    hand = cl.Hand()
    deck = cl.StandardDeck()
    hand.get_cards(deck, 5)

    assert len(hand.cards) == 5
    assert len(deck.cards) == 52 - len(hand.cards)


def test_suits():
    hand = cl.Hand()
    deck = cl.StandardDeck()
    hand.get_cards(deck, 1)
    suit = hand.cards[0].suit.name
    assert suit == 'Clubs'


''' Test Poker Hands '''


def test_straight_flush():
    deck = cl.StandardDeck()
    straight_flush_hand = cl.Hand()
    straight_flush_hand.get_cards(deck, 5)
    pokerhand = cl.PokerHand(straight_flush_hand.cards)
    assert pokerhand.card_value == 6
    assert pokerhand.poker_type == cl.PokerHandTypes.Straight_flush

    return pokerhand


def test_flush():
    deck = cl.StandardDeck()
    flush_hand = cl.Hand()
    for i in range(5):
        flush_hand.get_cards(deck, 1)
        deck.deal_cards()

    pokerhand = cl.PokerHand(flush_hand.cards)

    assert pokerhand.poker_type == cl.PokerHandTypes.Flush
    assert pokerhand.card_value == 10


def test_full_house():
    deck = cl.StandardDeck()
    full_house_hand = cl.Hand()
    deck.sort_cards()

    full_house_hand.get_cards(deck, 3)

    deck.deal_cards()

    full_house_hand.get_cards(deck, 2)

    pokerhand = cl.PokerHand(full_house_hand.cards)

    assert pokerhand.poker_type == cl.PokerHandTypes.Full_house
    assert pokerhand.card_value == 2

    return pokerhand


def test_four_of_a_kind():
    deck = cl.StandardDeck()
    fours_hand = cl.Hand()
    deck.sort_cards()

    fours_hand.get_cards(deck, 5)

    pokerhand = cl.PokerHand(fours_hand.cards)

    assert pokerhand.poker_type == cl.PokerHandTypes.Four_of_a_kind
    assert pokerhand.card_value == 2


def test_three_of_a_kind():
    deck = cl.StandardDeck()
    triple_hand = cl.Hand()
    deck.sort_cards()

    triple_hand.get_cards(deck, 3)
    for j in range(2):
        for i in range(3):
            deck.deal_cards()
        triple_hand.get_cards(deck, 1)

    pokerhand = cl.PokerHand(triple_hand.cards)

    assert pokerhand.poker_type == cl.PokerHandTypes.Three_of_a_kind
    assert pokerhand.card_value == 2

    return pokerhand


def test_one_pair():
    deck = cl.StandardDeck()
    pair_hand = cl.Hand()
    deck.sort_cards()

    pair_hand.get_cards(deck, 2)
    for j in range(3):
        for i in range(3):
            deck.deal_cards()
        pair_hand.get_cards(deck, 1)

    pokerhand = cl.PokerHand(pair_hand.cards)

    assert pokerhand.poker_type == cl.PokerHandTypes.One_pair
    assert pokerhand.card_value == 2


def test_two_pair():
    deck = cl.StandardDeck()
    two_pair_hand = cl.Hand()
    deck.sort_cards()

    two_pair_hand.get_cards(deck, 2)
    for j in range(1):
        for i in range(6):
            deck.deal_cards()
        two_pair_hand.get_cards(deck, 2)
        for i in range(6):
            deck.deal_cards()
        two_pair_hand.get_cards(deck, 1)

    pokerhand = cl.PokerHand(two_pair_hand.cards)

    assert pokerhand.poker_type == cl.PokerHandTypes.Two_pair
    assert pokerhand.card_value == 4

    return pokerhand


def test_compare_two_pair_threes():
    two_pair_hand = test_two_pair()
    triple_hand = test_three_of_a_kind()

    assert triple_hand == triple_hand
    assert triple_hand > two_pair_hand
    assert not triple_hand == two_pair_hand
    assert two_pair_hand < triple_hand


def test_compare_full_house_straight_flush():
    full_house = test_full_house()
    straight_flush = test_straight_flush()

    assert full_house < straight_flush


def test_drop_card_from_empty_hand():
    deck = cl.StandardDeck()
    hand = cl.Hand()
    hand.get_cards(deck, 5)

    print("\n Drop card on position 2, 3, 4")
    hand.drop_cards([2, 3, 4])
    for i in range(3):
        print("Drop my first card!")
        hand.drop_cards([0])


def test_deal_card_from_empty_deck():
    deck = cl.StandardDeck()
    for i in range(53):
        try:
            deck.deal_cards()
        except IndexError:
            print("Error will be sent if empty")


def test_straigth():
    deck = cl.StandardDeck()
    hand = cl.Hand()
    for i in range(12):
        deck.deal_cards()
    hand.get_cards(deck, 4)
    for i in range(13):
        deck.deal_cards()
    hand.get_cards(deck, 1)
    pokerhand = cl.PokerHand(hand.cards)
    print(hand)
    assert pokerhand.poker_type == cl.PokerHandTypes.Straight
    assert pokerhand.card_value == 14


def test_best_poker_hand():
    board = cl.Hand()
    deck = cl.StandardDeck()

    hand = cl.Hand()
    board.get_cards(deck, 5)
    hand.get_cards(deck, 2)

    best_hand = hand.best_poker_hand(board.cards)

    assert best_hand.poker_type == cl.PokerHandTypes.Straight_flush
    assert best_hand.card_value == 8
