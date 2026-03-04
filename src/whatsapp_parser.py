"""
WhatsApp Export Parser Module
Week 1-2: Parse WhatsApp messages
"""

import re
from collections import Counter
import json


class WhatsAppParser:
    """Parse WhatsApp export files"""
    
    def __init__(self, file_path):
        """Initialize parser"""
        self.file_path = file_path
        self.messages = []
        self.user_mapping = {}
        self.next_user_id = 1
    
    def parse_line(self, line):
        """Parse a single WhatsApp message line"""
        # TODO: Implement in Week 1-2
        # Pattern: [Date, Time] Sender: Message
        pass
    
    def parse_file(self):
        """Parse entire WhatsApp export file"""
        # TODO: Implement in Week 1-2
        pass
    
    def get_statistics(self):
        """Get basic statistics"""
        # TODO: Implement in Week 1-2
        pass
    
    def save_to_json(self, output_file):
        """Save parsed messages to JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, indent=2, ensure_ascii=False)