import sys
from DAT171_Python_assignments.CA3.card_view import *
from DAT171_Python_assignments.CA3 import texas_holdem

# Every Qt application must have one and only one QApplication object;
# it receives the command line arguments passed to the script, as they
# can be used to customize the application's appearance and behavior
qt_app = QApplication(sys.argv)


class StatusView(QGroupBox):
    """ Object that show which players turn it is. """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.new_current_player.connect(self.update)

        self.active_player = QLabel('Active Player: ')

        status_layout = QHBoxLayout()
        status_layout.addWidget(self.active_player)

        self.setLayout(status_layout)

        self.update()

    def update(self):
        self.active_player.setText('Active Player: {}'.format(self.model.get_active_player().name))


class Controllers(QGroupBox):
    """ Class that shows all the controllers that a player can use. """
    def __init__(self, model):
        super().__init__("Controllers!")  # Call the QWidget initialization as well!
        self.model = model

        pot = PotView(self.model.pot) # Define the pot as an object

        # Controller buttons
        check_button = QPushButton("Check")
        raise_button = QPushButton("Raise")
        bet_unit = QLabel('$')
        bet_amount = QSpinBox()
        fold = QPushButton("Fold")
        all_in = QPushButton("All in!")

        def raise_bet():
            self.model.bet(bet_amount.value())

        check_button.clicked.connect(self.model.call)
        raise_button.clicked.connect(raise_bet)
        fold.clicked.connect(self.model.fold)
        all_in.clicked.connect(self.model.all_in)

        """ Format the widgets in a nice order. """
        hbox_raise = QHBoxLayout()
        hbox_raise.addWidget(raise_button)
        hbox_raise.addWidget(bet_unit)
        hbox_raise.addWidget(bet_amount)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(pot)
        vbox.addWidget(check_button)
        vbox.addWidget(fold)
        vbox.addWidget(all_in)
        vbox.addLayout(hbox_raise)

        # Set Layout
        self.setLayout(vbox)


class TableView(QGroupBox):
    """ Class that contain up to five cards and show them on the table. """
    def __init__(self, model: texas_holdem.TexasHoldemMode):
        super().__init__('Table')
        self.model = model
        self.table_cards = CardView(self.model.table_cards)

        # Layout for the table in the window.
        card_layout = QHBoxLayout()
        card_layout.addWidget(self.table_cards)

        main_layout = QVBoxLayout()
        main_layout.addLayout(card_layout)

        self.setLayout(main_layout)


class PotView(QLabel):
    """ An object that shows how much credits we have in the pot for the players. """
    def __init__(self, pot: texas_holdem.PotModel):
        super().__init__()
        pot.new_value.connect(lambda: self.setText('${}'.format(pot.credits)))


class PlayerView(QGroupBox):
    """ An object that shows how much a player is betting, how much credit a player have and the name of the player. """
    def __init__(self, model: texas_holdem.Player):
        super().__init__(model.name)

        self.model = model
        player_cards = CardView(self.model.hand)

        self.chips = QLabel()
        self.chips.setAlignment(Qt.AlignBottom)

        # Layout
        card_layout = QHBoxLayout()
        card_layout.addWidget(player_cards)
        card_layout.addWidget(self.chips)
        self.setLayout(card_layout)

        self.model.new_credits.connect(self.update_chips)

        self.update_chips()

    def update_chips(self):
        self.chips.setText('${}'.format(self.model.credits))


class WindowView(QWidget):
    """ The main window grouping all other views in a good way. """
    def __init__(self, model):
        super().__init__()
        self.model = model

        status = StatusView(self.model)
        control = Controllers(self.model)
        table = TableView(self.model)

        players = [PlayerView(player) for player in self.model.players]

        horizontal = QHBoxLayout()
        horizontal.addWidget(players[0])
        horizontal.addWidget(control)
        horizontal.addWidget(players[1])

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(status)
        main_layout.addWidget(table)
        main_layout.addLayout(horizontal)

        self.setLayout(main_layout)
        self.setGeometry(200, 300, 1300, 500) # Define window size

        # Connect logic
        self.model.player_win.connect(self.player_wins)

    def player_wins(self):
        """ Calling the winner window """
        winner_view = WinnerView(self.model)
        winner_view.show()


class WinnerView(QWidget):
    """ Popup window that is called upon if game is over """

    def __init__(self, model):
        super().__init__()

        self.model = model

        self.setWindowTitle("GAME ENDED")
        self.setGeometry(650, 450, 500, 500)
        reply = QMessageBox.question(self, "WINNER:", 'Winning player: {}, with hand: {}'
                                     .format(self.model.players[self.model.winning_player_index].name,
                                             self.model.winning_hand.poker_type.name, QMessageBox.Ok, QMessageBox.Ok))


