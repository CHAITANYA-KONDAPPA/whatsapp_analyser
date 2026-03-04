#!/usr/bin/env python3
"""
WhatsApp Chat Analyzer - Command Line Interface
"""

import sys
import os
from src.pipeline import ChatAnalysisPipeline


def main():
    """Main CLI entry point"""
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <whatsapp_export.txt>")
        print("\nExample: python main.py chat.txt")
        sys.exit(1)
    
    export_file = sys.argv[1]
    
    if not os.path.exists(export_file):
        print(f"❌ Error: File '{export_file}' not found")
        sys.exit(1)
    
    print("=" * 60)
    print("WhatsApp Chat Analysis Pipeline")
    print("=" * 60)
    
    try:
        # Run pipeline
        pipeline = ChatAnalysisPipeline(export_file)
        success = pipeline.run_full_pipeline()
        
        if success:
            print("\n" + "=" * 60)
            print("✅ ANALYSIS COMPLETE!")
            print("=" * 60)
            print("\n📊 Results:")
            print("  - sentiment_analyzed.csv")
            print("  - visualizations/")
            print("  - statistics.json")
            sys.exit(0)
        else:
            print("\n❌ Analysis failed")
            sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()