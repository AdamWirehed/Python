from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtSvg import *

import sys

from Assignment3.CA3_Gabriel.texasholdem import *


class PlayerView(QGroupBox):
    """ The view of each of the two players, with resp. name, cards and current money """
    def __init__(self, name, player_model):
        super().__init__(name)
        self.model = player_model

        # Name
        self.player_name = QLabel(self.model.name)
        self.player_name.setAlignment(Qt.AlignCenter)

        # Cards on hand
        self.player_cards = CardView(self.model.hand)
        self.player_cards.setAlignment(Qt.AlignCenter)

        # Money
        self.player_money = QLabel("$ 0")
        self.player_money.setAlignment(Qt.AlignCenter)

        # Main layout
        main_vbox = QVBoxLayout()
        main_vbox.addWidget(self.player_name)
        main_vbox.addStretch(1)
        main_vbox.addWidget(self.player_cards)
        main_vbox.addStretch(1)
        main_vbox.addWidget(self.player_money)

        self.setLayout(main_vbox)

        # Connect logic
        self.model.new_player_data.connect(self.update)

        self.update()

    def update(self):
        """ If the money is changed, the view is to be updated """
        self.player_money.setText("$ {}".format(self.model.money))


class ControlView(QGroupBox):
    """ The three controller buttons the players are to use """
    def __init__(self, model):
        super().__init__("CONTROLLERS")

        # Control buttons

        self.fold_button = QPushButton("Fold")
        self.call_button = QPushButton("Call")
        self.raise_button = QPushButton("Raise")

        button_box = QVBoxLayout()
        button_box.addStretch(1)
        button_box.addWidget(self.fold_button)
        button_box.addStretch(1)
        button_box.addWidget(self.call_button)
        button_box.addStretch(1)
        button_box.addWidget(self.raise_button)
        button_box.addStretch(1)

        # Layout
        self.setLayout(button_box)

        # Connect logic

        self.model = model
        self.fold_button.clicked.connect(self.model.fold)
        self.call_button.clicked.connect(self.model.call)
        self.raise_button.clicked.connect(self.raise_value)

    def raise_value(self):
        """ When the raise button is clicked, open input dialog for to receive amount of the bet """
        active_player_money = self.model.players[self.model.active_player].money
        non_active_player_money = self.model.players[not self.model.active_player].money
        int, okpressed = QInputDialog.getInt(self, "Give integer raise amount", "Amount:", 20, 0,
                                             min([active_player_money-self.model.last_bet,
                                                  non_active_player_money]), 2)
        if okpressed:
            self.model.raise_bet(int)


class TableView(QGroupBox):
    """ The view of the cards of the table and the pot size """
    def __init__(self, table_hand_model):
        super().__init__()
        self.model = table_hand_model

        # cards
        self.table_hand = self.model.table
        self.table_cards = CardView(self.table_hand)
        self.table_hand.flip()
        # TODO: CENTER THE CARDS

        # Pot
        self.pot = QLabel()

        # Pot layout
        pot_layout = QHBoxLayout()
        pot_layout.addStretch(1)
        pot_layout.addWidget(self.pot)
        pot_layout.addStretch(1)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.table_cards)
        main_layout.addLayout(pot_layout)

        self.setLayout(main_layout)

        # Connect logic
        self.model = table_hand_model
        self.table_hand = self.model.table
        self.model.new_table_data.connect(self.update)

        self.update()

    def update(self):
        self.pot.setText("$ {}".format(self.model.pot))
        self.table_cards.change_cards()


class StatusBar(QGroupBox):
    """
    Top bar of the game window, with the status of the game incl. rounds and active player.
    Also a button to stop the game.
    """
    def __init__(self, model):
        super().__init__()

        # Buttons
        self.stop_button = QPushButton("STOP")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.stop_button)

        # Titles
        title = QLabel("TEXAS HOLD 'EM")
        self.round = QLabel("ROUND: XX")
        self.active_player = QLabel("Active player: ")

        # Layout
        status_layout = QHBoxLayout()
        status_layout.addWidget(title)
        status_layout.addStretch(1)
        status_layout.addWidget(self.round)
        status_layout.addStretch(1)
        status_layout.addWidget(self.active_player)
        status_layout.addStretch(1)
        status_layout.addLayout(button_layout)

        self.setLayout(status_layout)

        # Connect logic
        self.model = model
        self.model.new_status_data.connect(self.update)
        self.stop_button.clicked.connect(self.model.stop_game)

        self.update()

    def update(self):
        self.round.setText("ROUND: {}".format(self.model.round))
        self.active_player.setText("ACTIVE PLAYER: {}".format(self.model.players[self.model.active_player].name))


class WindowView(QGroupBox):
    """ The full window with the playable game """
    def __init__(self, game_model):
        super().__init__()
        self.model = game_model

        # Initializing players
        player1 = PlayerView("Player 1", self.model.players[0])
        player2 = PlayerView("Player 2", self.model.players[1])

        # Calling the other classes the window consists of
        control = ControlView(self.model)
        table = TableView(self.model)
        status = StatusBar(self.model)

        # Bottom layout with: Player, Controllers, Player
        bottom = QHBoxLayout()
        bottom.addWidget(player1)
        bottom.addWidget(control)
        bottom.addWidget(player2)

        # main
        main_layout = QVBoxLayout()
        main_layout.addWidget(status)
        main_layout.addWidget(table)
        main_layout.addLayout(bottom)

        self.setLayout(main_layout)
        self.setGeometry(200, 200, 1200, 800)

        # Connect
        self.model.game_ended.connect(self.game_end)

    def game_end(self):
        """ Calling the window """
        winner_view = WinnerView(self.model)
        winner_view.show()


class WinnerView(QWidget):
    """ Popup window that is called upon if game is over """

    def __init__(self, game_model):
        super().__init__()

        self.model = game_model

        if self.model.players[0].money > self.model.players[1].money:
            self.winner = self.model.players[0]
            self.loser = self.model.players[1]
            text = "WINNER: {}, with ${}".format(self.winner.name, self.winner.money)

        elif self.model.players[0].money < self.model.players[1].money:
            self.winner = self.model.players[1]
            self.loser = self.model.players[0]
            text = "WINNER: {}, with ${}".format(self.winner.name, self.winner.money)

        else:
            text = "GAME DRAW"

        self.setWindowTitle("GAME ENDED")
        self.setGeometry(650, 450, 300, 300)
        reply = QMessageBox.question(self, "GAME ENDED:", text, QMessageBox.Ok, QMessageBox.Ok)

        if reply == QMessageBox.Ok:
            self.model.quit_game()


class CardView(QGraphicsView):
    """ A View widget that represents the table area displaying a players cards. """

    # Underscores indicate a private function/method!
    def __read_cards():
        """
        Reads all the 52 cards from files.
        :return: Dictionary of SVG renderers
        """
        all_cards = dict()  # Dictionaries let us have convenient mappings between cards and their images
        for suit_file, suit in zip('HDSC', range(4)):
            for value_file, value in zip(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'],
                                         range(2, 15)):
                file = value_file + suit_file
                key = (value, suit)  # I'm choosing this tuple to be the key for this dictionary
                all_cards[key] = QSvgRenderer('cards/' + file + '.svg')
        return all_cards

    # We read all the card graphics as static class variables
    back_card = QSvgRenderer('cards/Red_Back_2.svg')
    all_cards = __read_cards()

    def __init__(self, model, card_spacing=250, padding=10):
        """
        Initializes the view to display the content of the given model
        :param model: A model that represents a set of cards.
        The model should have: data_changed, cards, clicked_position, flipped,
        :param card_spacing: Spacing between the visualized cards.
        :param padding: Padding of table area around the visualized cards.
        """
        self.scene = TableScene()
        super().__init__(self.scene)

        self.model = model
        self.card_spacing = card_spacing
        self.padding = padding

        # Whenever the this window should update, it should call the "change_cards" method.
        # This can, for example, be done by connecting it to a signal.
        # The view can listen to changes:
        model.data_changed.connect(self.change_cards)
        # It is completely optional if you want to do it this way, or have some overreaching Player/GameState
        # call the "change_cards" method instead. z

        # Add the cards the first time around to represent the initial state.
        self.change_cards()

    def change_cards(self):
        """ If the cards are changed, this method is called upon to render the new ones """
        # Add the cards from scratch
        self.scene.clear()
        for i, card in enumerate(self.model.cards):
            # The ID of the card in the dictionary of images is a tuple with (value, suit), both integers
            graphics_key = (card.get_value(), card.get_suit().value)
            renderer = self.back_card if self.model.flipped(i) else self.all_cards[graphics_key]
            c = CardItem(renderer, i)

            # Shadow effects
            shadow = QGraphicsDropShadowEffect(c)
            shadow.setBlurRadius(10.)
            shadow.setOffset(5, 5)
            shadow.setColor(QColor(0, 0, 0, 180))  # Semi-transparent black!
            c.setGraphicsEffect(shadow)

            # Place the cards on the default positions
            c.setPos(c.position * self.card_spacing, 0)
            self.scene.addItem(c)

        self.update_view()

    def update_view(self):
        """ This method updates the view of the cards, if they are changed """
        scale = (self.viewport().height()-2*self.padding)/313
        self.resetTransform()
        self.scale(scale, scale)
        # Put the scene bounding box
        self.setSceneRect(-self.padding//scale, -self.padding//scale,
                          self.viewport().width()//scale, self.viewport().height()//scale)

    def resizeEvent(self, painter):
        """ This method is called when the window is resized """
        # If the widget is resize, we gotta adjust the card sizes.
        # QGraphicsView automatically re-paints everything when we modify the scene.
        self.update_view()
        super().resizeEvent(painter)

    def mouseDoubleClickEvent(self, event):
        """ If the cards of a view is double clicked, they are to flip"""
        self.model.flip()  # Another possible event. Lets add it to the flip functionality for fun!


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

