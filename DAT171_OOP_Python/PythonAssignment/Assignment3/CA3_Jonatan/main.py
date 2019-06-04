
from DAT171_Python_assignments.CA3.gui import*
from DAT171_Python_assignments.CA3.texas_holdem import*


qt_app = QApplication(sys.argv)  # Starting application

model = TexasHoldemMode(["Adam", "Jonatan"], 100)  # Input data for the game: Names and buyin

w = WindowView(model)  # Calling the game view with the created model

w.show()

qt_app.exec_()