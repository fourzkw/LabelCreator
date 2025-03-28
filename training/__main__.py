import sys
from PyQt5.QtWidgets import QApplication
from .trainer_ui import YoloTrainerUI

def main():
    app = QApplication(sys.argv)
    window = YoloTrainerUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()