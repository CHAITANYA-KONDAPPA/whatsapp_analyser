"""
Main Analysis Pipeline
Week 7: Combine all modules
"""

from .whatsapp_parser import WhatsAppParser
from .data_processor import DataProcessor
from .nlp_processor import NLPProcessor
from .sentiment_analyzer import SentimentAnalyzer
from .visualizer import ChatVisualizer


class ChatAnalysisPipeline:
    """Complete analysis pipeline"""
    
    def __init__(self, whatsapp_export_path):
        """Initialize pipeline"""
        self.export_path = whatsapp_export_path
        self.df = None
    
    def run_full_pipeline(self):
        """Run complete pipeline from start to finish"""
        # TODO: Implement in Week 7
        # Should call: parse → process → nlp → sentiment → visualize
        pass