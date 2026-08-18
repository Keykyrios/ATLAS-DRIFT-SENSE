from PyQt6.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
from src.atlas.pipeline import ATLASPipeline
from src.atlas.types import ATLASResult

class PipelineWorker(QThread):
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    
    def __init__(self, ref_path: str, search_path: str):
        super().__init__()
        self.ref_path = ref_path
        self.search_path = search_path
        self.pipeline = ATLASPipeline()
        
    def run(self):
        try:
            ref_img = cv2.imread(self.ref_path, cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(self.search_path, cv2.IMREAD_GRAYSCALE)
            
            if ref_img is None or search_img is None:
                self.error_signal.emit("Failed to load images")
                return
                
            result = self.pipeline.process(ref_img, search_img)
            self.finished_signal.emit(result)
        except Exception as e:
            self.error_signal.emit(str(e))
