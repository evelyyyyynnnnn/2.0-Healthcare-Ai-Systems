#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression test for the empathy word cloud generation fix."""

import sys
import os
# Add src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from empathy_analysis import EnhancedEmpathyAnalyzer
import json

def test_wordcloud_generation():
    """Exercise the word cloud generation workflow."""
    print("🔧 Testing word cloud generation...")
    
    # Instantiate the analyzer
    analyzer = EnhancedEmpathyAnalyzer()
    
    # Load existing analysis results
    try:
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'detailed_empathy_analysis.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        analysis_results = [data]  # Wrap the full JSON payload in a list
        print(f"✅ Loaded analysis results with {data.get('total_cases', 0)} case(s)")
    except Exception as e:
        print(f"❌ Failed to load analysis results: {e}")
        return False
    
    # Execute the word cloud generation workflow
    try:
        print("\n📊 Generating word cloud...")
        wordcloud = analyzer.generate_wordcloud(analysis_results)
        
        if wordcloud is not None:
            print("✅ Word cloud generated successfully.")
            print("📁 Saved image: outputs/figures/enhanced_empathy_keywords_wordcloud.png")
            return True
        else:
            print("❌ Word cloud generation failed.")
            return False
            
    except Exception as e:
        print(f"❌ Exception during word cloud generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Entry point for the word cloud regression test."""
    print("🚀 Starting word cloud regression test...")
    
    # Ensure the output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute the test
    success = test_wordcloud_generation()
    
    if success:
        print("\n🎉 Word cloud regression test passed.")
    else:
        print("\n💥 Test failed. Check the logs for details.")
    
    return success

if __name__ == "__main__":
    main() 