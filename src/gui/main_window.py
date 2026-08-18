from PyQt6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget, QLabel
from PyQt6.QtCore import Qt

class ATLASMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ATLAS / Drift-Sense Console")
        self.resize(1200, 800)
        
        self.tabs = QTabWidget()
        
        # 1. Dataset Studio
        self.dataset_tab = QWidget()
        self.dataset_layout = QVBoxLayout()
        self.dataset_layout.addWidget(QLabel("Dataset Studio (Configure Synthetic Data Generation)"))
        self.dataset_tab.setLayout(self.dataset_layout)
        
        # 2. Pipeline Console
        self.pipeline_tab = QWidget()
        self.pipeline_layout = QVBoxLayout()
        self.pipeline_layout.addWidget(QLabel("Pipeline Console (Run ATLAS on a single pair)"))
        self.pipeline_tab.setLayout(self.pipeline_layout)
        
        # 3. Batch Evaluation
        self.batch_tab = QWidget()
        self.batch_layout = QVBoxLayout()
        self.batch_layout.addWidget(QLabel("Batch Evaluation Dashboard"))
        self.batch_tab.setLayout(self.batch_layout)
        
        self.tabs.addTab(self.dataset_tab, "1. Dataset Studio")
        self.tabs.addTab(self.pipeline_tab, "2. Pipeline Console")
        self.tabs.addTab(self.batch_tab, "3. Batch Dashboard")
        
        self.setCentralWidget(self.tabs)
