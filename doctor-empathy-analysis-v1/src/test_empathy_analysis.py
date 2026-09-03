#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test harness for the doctor empathy analysis system."""

import sys
import os
sys.path.append('src')

from empathy_analysis import EmpathyAnalyzer

def test_empathy_recognition():
    """Validate empathy keyword recognition."""
    print("=== Empathy Recognition Test ===\n")
    
    # Instantiate the analyzer
    analyzer = EmpathyAnalyzer()
    
    # Sample texts (based on real consultation data)
    test_texts = [
        "感谢您的信任，病情资料我已详细阅读。您目前的检查资料不齐全，需要补充以下检查：既往史",
        "能够理解，不行只能手术解决",
        "不要太着急，小孩毕竟还小很多事情呢，慢慢来啊，反正就要一个是要重视啊，好吧",
        "密切观察小儿的视功能发育，定期复查",
        "建议完善眼肌MRI及甲功检查",
        "注意休息，局部热敷按摩",
        "确实，这个情况需要重视",
        "我明白您的担心，但是不用太紧张",
        "建议您定期到医院检查，看看他那个白内障的那个变化情况",
        "您的眼睛原来有过什么其他情况"
    ]
    
    print("Sample texts:")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    
    print("\n=== Empathy Feature Detection Results ===\n")
    
    # Evaluate each sample
    for i, text in enumerate(test_texts, 1):
        print(f"Text {i}: {text}")
        print("-" * 50)
        
        # Detect empathy keywords
        empathy_words = analyzer._identify_empathy_words(text)
        
        if empathy_words:
            print(f"Detected empathy keywords: {empathy_words}")
            
            # Group keywords by empathy category
            categorized_words = {}
            for word in empathy_words:
                for category, features in analyzer.empathy_features.items():
                    if word in features:
                        if category not in categorized_words:
                            categorized_words[category] = []
                        categorized_words[category].append(word)
                        break
            
            print("By category:")
            for category, words in categorized_words.items():
                print(f"  {category}: {words}")
        else:
            print("No empathy keywords detected.")
        
        print()
    
    # Test word cloud generation
    print("=== Word Cloud Test ===\n")
    
    # Create mock analysis results
    mock_analysis_results = {
        'consultations': [
            {'doctor_speech': ' '.join(test_texts[:5])},
            {'doctor_speech': ' '.join(test_texts[5:])}
        ]
    }
    
    try:
        result = analyzer.generate_wordcloud(mock_analysis_results)
        if result:
            print(f"Word cloud generated: {result}")
        else:
            print("Word cloud generation failed.")
    except Exception as e:
        print(f"Word cloud generation error: {e}")
    
    print("\n=== Test Finished ===")

if __name__ == "__main__":
    test_empathy_recognition()
