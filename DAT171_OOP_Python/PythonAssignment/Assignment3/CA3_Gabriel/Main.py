from Assignment3.CA3_Gabriel.texasholdem import *
from Assignment3.CA3_Gabriel.gui import *

""" Main code to start a game of TEXAS HOLD 'EM """

qt_app = QApplication(sys.argv)  # Starting application

model = TexasHoldEmModel(["Gabriel", "Oliver"], 100)  # Input data for the game: Names and buyin

w = WindowView(model)  # Calling the game view with the created model

w.show()

qt_app.exec_()
