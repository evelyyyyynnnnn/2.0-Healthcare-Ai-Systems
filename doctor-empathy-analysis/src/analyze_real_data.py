#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyze empathy expression in real-world medical consultations."""

import sys
import os
import pandas as pd
import re
sys.path.append('src')

from empathy_analysis import EmpathyAnalyzer

def extract_doctor_speech_from_data(df):
    """Extract doctor utterances from the dataframe."""
    doctor_speeches = []
    
    for idx, row in df.iterrows():
        # Identify the dialogue column (defaults to the 5th column)
        conversation_col = df.columns[4] if len(df.columns) > 4 else df.columns[-1]
        conversation_text = str(row[conversation_col])
        
        # Extract speech segments that begin with the doctor's name
        doctor_pattern = r'陈锦昌副主任医师:([^患者]*?)(?=患者:|$)'
        matches = re.findall(doctor_pattern, conversation_text)
        
        for match in matches:
            # Clean up the text
            cleaned_text = re.sub(r'\(\d{4}-\d{2}-\d{2}\)', '', match)  # Remove date stamps
            cleaned_text = re.sub(r'以上文字由机器转写，仅供参考', '', cleaned_text)  # Remove transcription notice
            cleaned_text = re.sub(r'\d+["″]', '', cleaned_text)  # Remove time markers
            cleaned_text = cleaned_text.strip()
            
            if cleaned_text and len(cleaned_text) > 5:  # Skip extremely short snippets
                doctor_speeches.append(cleaned_text)
    
    return doctor_speeches

def analyze_empathy_patterns(doctor_speeches):
    """Analyze empathy expression patterns."""
    analyzer = EmpathyAnalyzer()
    
    print("=== Empathy analysis for doctor speech ===\n")
    
    total_speeches = len(doctor_speeches)
    empathy_speeches = 0
    empathy_word_counts = {}
    category_counts = {}
    
    for i, speech in enumerate(doctor_speeches, 1):
        print(f"Utterance {i}: {speech}")
        print("-" * 60)
        
        # Identify empathy keywords
        empathy_words = analyzer._identify_empathy_words(speech)
        
        if empathy_words:
            empathy_speeches += 1
            print(f"✓ Detected empathy keywords: {empathy_words}")
            
            # Group by empathy category
            categorized_words = {}
            for word in empathy_words:
                for category, features in analyzer.empathy_features.items():
                    if word in features:
                        if category not in categorized_words:
                            categorized_words[category] = []
                        categorized_words[category] = categorized_words.get(category, []) + [word]
                        
                        # Track category frequency
                        category_counts[category] = category_counts.get(category, 0) + 1
                        break
            
            print("By category:")
            for category, words in categorized_words.items():
                print(f"  {category}: {words}")
                
                # Track keyword frequency
                for word in words:
                    empathy_word_counts[word] = empathy_word_counts.get(word, 0) + 1
        else:
            print("✗ No empathy keywords detected.")
        
        print()
    
    # Summary statistics
    print("=== Summary ===")
    print(f"Total utterances: {total_speeches}")
    print(f"Utterances with empathy: {empathy_speeches}")
    print(f"Empathy ratio: {empathy_speeches/total_speeches*100:.1f}%")
    
    print("\nEmpathy keyword frequency:")
    sorted_words = sorted(empathy_word_counts.items(), key=lambda x: x[1], reverse=True)
    for word, count in sorted_words:
        print(f"  {word}: {count}")
    
    print("\nEmpathy category frequency:")
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for category, count in sorted_categories:
        print(f"  {category}: {count}")
    
    return empathy_word_counts, category_counts

def main():
    """Entry point for analyzing real consultation data."""
    print("=== Empathy analysis for medical consultations ===\n")
    
    try:
        # Load the dataset
        print("Loading data...")
        df = pd.read_excel('data/Sample Data.xlsx')
        print(f"Loaded {len(df)} rows.")
        
        # Extract doctor speech
        print("\nExtracting doctor speech...")
        doctor_speeches = extract_doctor_speech_from_data(df)
        print(f"Extracted {len(doctor_speeches)} utterances.")
        
        if not doctor_speeches:
            print("No doctor speech found. Please verify the data format.")
            return
        
        # Analyze empathy patterns
        empathy_words, category_counts = analyze_empathy_patterns(doctor_speeches)
        
        # Generate the word cloud
        print("\nGenerating word cloud...")
        analyzer = EmpathyAnalyzer()
        
        # Construct analysis result payload
        analysis_results = {
            'consultations': [
                {'doctor_speech': speech} for speech in doctor_speeches
            ]
        }
        
        result = analyzer.generate_wordcloud(analysis_results)
        if result:
            print(f"Word cloud generated: {result}")
        else:
            print("Word cloud generation failed.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
