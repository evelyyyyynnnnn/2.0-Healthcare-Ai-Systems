#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test suite for the enhanced empathy analyzer."""

import sys
import os
# Add src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from empathy_analysis import EnhancedEmpathyAnalyzer
import numpy as np

def test_basic_functionality():
    """Ensure core feature extraction works."""
    print("\n🔧 Testing core functionality...")
    
    analyzer = EnhancedEmpathyAnalyzer()
    
    # Validate feature extraction
    test_text = "我理解您的担心，这种症状确实会让人焦虑"
    features = analyzer.extract_enhanced_features(test_text)
    empathy_features = analyzer.extract_empathy_features(test_text)
    
    print(f"Enhanced feature count: {len(features)}")
    print(f"Empathy feature count: {len(empathy_features)}")
    
    return len(features) > 0 and len(empathy_features) > 0

def test_enhanced_excel_functionality():
    """Exercise the enhanced Excel analysis workflow."""
    print("\n📊 Testing enhanced Excel analysis...")
    
    analyzer = EnhancedEmpathyAnalyzer()
    
    # Validate doctor speech extraction
    test_conversation = "陈锦昌副主任医师:我理解您的担心(2024.01.01) 患者:我很焦虑 陈锦昌副主任医师:别担心，会好的(2024.01.01)"
    doctor_speech = analyzer.extract_doctor_speech(test_conversation)
    
    print(f"Extracted doctor speech: {doctor_speech}")
    
    # Compare enhanced empathy scoring against baseline
    test_text = "我理解您的担心，这种症状确实会让人焦虑"
    enhanced_result = analyzer.calculate_enhanced_empathy_score(test_text, use_enhanced_keywords=True)
    original_result = analyzer.calculate_enhanced_empathy_score(test_text, use_enhanced_keywords=False)
    
    print(f"Enhanced total score: {enhanced_result['total_score']:.2f}")
    print(f"Baseline total score: {original_result['total_score']:.2f}")
    print(f"Category scores: {enhanced_result['category_scores']}")
    
    # Validate empathy_scores.csv export
    print("\n📋 Testing empathy_scores.csv export...")
    
    # Build mock analysis output
    mock_analysis_results = [
        {
            'case_id': 'Case_001',
            'doctor_name': '陈锦昌副主任医师',
            'patient_age': '45',
            'patient_gender': '女',
            'disease_category': '内科疾病',
            'consultation_date': '2024.01.01',
            'avg_empathy_score': 8.5,
            'empathy_density': 0.85,
            'word_count': 20,
            'dialogue_length': 50,
            'empathy_features_count': 5,
            'empathy_scores': {
                '感谢信任': 1.0,
                '理解认同': 2.0,
                '关心注意': 1.5,
                '安慰支持': 2.5,
                '倾听确认': 1.0,
                '耐心解释': 2.0
            },
            'doctor_speech': '我理解您的担心，这种症状确实会让人焦虑，别担心，会好的'
        }
    ]
    
    # Invoke export functionality
    success = analyzer.export_empathy_scores_csv(mock_analysis_results, 'test_empathy_scores.csv')
    
    if success:
        print("✅ export_empathy_scores_csv succeeded")
        # Clean up the test artifact
        import os
        if os.path.exists('test_empathy_scores.csv'):
            os.remove('test_empathy_scores.csv')
            print("🧹 Removed temporary CSV")
    else:
        print("❌ export_empathy_scores_csv failed")
    
    return (len(doctor_speech) > 0 and 
            enhanced_result['total_score'] > 0 and 
            original_result['total_score'] > 0 and
            success)

def test_ml_functionality():
    """Validate the machine learning pipeline."""
    print("\n🤖 Testing machine learning components...")
    
    analyzer = EnhancedEmpathyAnalyzer()
    
    # Create training data
    training_data = analyzer.create_synthetic_training_data()
    X, y = analyzer.prepare_training_data(training_data)
    
    print(f"Training feature shape: {X.shape}")
    print(f"Training label distribution: {np.sum(y, axis=0)}")
    
    # Cross-validation smoke test
    print("\nRunning cross-validation...")
    cv_results = analyzer.cross_validate_models(X, y, cv_folds=3)
    
    for model_name, results in cv_results.items():
        print(f"{model_name}: CV F1 = {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
    
    # Train the models and capture metrics
    print("\nTraining machine learning models...")
    ml_results = analyzer.train_ml_models(X, y)
    
    for model_name, results in ml_results.items():
        print(f"{model_name}: F1 Micro = {results['f1_micro']:.3f}, F1 Macro = {results['f1_macro']:.3f}")
    
    # Run predictions against a sample text
    test_text = "我理解您的担心，这种症状确实会让人焦虑"
    print(f"\nPrediction sample: {test_text}")
    
    for model_name in ml_results.keys():
        try:
            result = analyzer.predict_empathy_ml(test_text, model_name)
            print(f"{model_name} prediction: empathy score = {result['empathy_score']:.3f}")
        except Exception as e:
            print(f"{model_name} prediction failed: {e}")
    
    # Feature importance analysis
    print("\nAnalyzing feature importance...")
    importance_df = analyzer.analyze_feature_importance('RandomForest')
    if importance_df is not None:
        print("Feature importance analysis complete.")
        print("Top 5 features:")
        for i, row in importance_df.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Ensemble prediction check
    print("\nTesting ensemble prediction...")
    try:
        ensemble_result = analyzer.ensemble_prediction(test_text)
        print(f"Ensemble empathy score: {ensemble_result['empathy_score']:.3f}")
        print("Ensemble predictions:")
        for label, pred in ensemble_result['ensemble_predictions'].items():
            prob = ensemble_result['ensemble_probabilities'].get(label, 0)
            print(f"  {label}: {'Yes' if pred else 'No'} (probability: {prob:.3f})")
    except Exception as e:
        print(f"Ensemble prediction failed: {e}")
    
    return True

def test_model_persistence():
    """Verify model persistence and reload."""
    print("\n💾 Testing model persistence...")
    
    analyzer = EnhancedEmpathyAnalyzer()
    
    # Create training data and train models
    training_data = analyzer.create_synthetic_training_data()
    X, y = analyzer.prepare_training_data(training_data)
    
    # Train the models
    ml_results = analyzer.train_ml_models(X, y)
    
    # Save models
    print("Saving models...")
    save_success = analyzer.save_models('test_models')
    
    if save_success:
        # Instantiate a fresh analyzer
        new_analyzer = EnhancedEmpathyAnalyzer()
        
        # Load serialized models
        print("Loading models...")
        load_success = new_analyzer.load_models('test_models')
        
        if load_success:
            # Evaluate the loaded model
            test_text = "我理解您的担心，这种症状确实会让人焦虑"
            try:
                result = new_analyzer.predict_empathy_ml(test_text, 'RandomForest')
                print(f"Loaded model prediction succeeded: empathy score = {result['empathy_score']:.3f}")
                return True
            except Exception as e:
                print(f"Loaded model prediction failed: {e}")
                return False
        else:
            print("Model loading failed.")
            return False
    else:
        print("Model save failed.")
        return False

def main():
    """Primary entry point for the enhanced analyzer tests."""
    print("🧪 Starting enhanced empathy analyzer tests...")
    print("=" * 60)
    
    tests = [
        ("Core functionality", test_basic_functionality),
        ("Enhanced Excel workflow", test_enhanced_excel_functionality),
        ("Machine learning pipeline", test_ml_functionality),
        ("Model persistence", test_model_persistence)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            if result:
                print(f"✅ {test_name} passed")
                passed_tests += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} raised an exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("📊 Test summary:")
    print(f"Total: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Please review the logs.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
