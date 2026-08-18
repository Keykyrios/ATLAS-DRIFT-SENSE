import sys
import os

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PyQt6.QtWidgets import QApplication
from src.gui.main_window import ATLASMainWindow

def main():
    app = QApplication(sys.argv)
    
    # Load QSS Theme
    qss_path = os.path.join(os.path.dirname(__file__), 'theme', 'style.qss')
    if os.path.exists(qss_path):
        with open(qss_path, 'r') as f:
            app.setStyleSheet(f.read())
            
    window = ATLASMainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
