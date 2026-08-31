#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the full automated test suite."""

import sys
import os
import subprocess
import time

def run_test(test_file):
    """Execute a single test module."""
    print(f"\n{'='*60}")
    print(f"🧪 Running: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Execute the test module
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Passed: {test_file}")
            print(f"⏱️  Duration: {duration:.2f}s")
            if result.stdout:
                print("📤 Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ Failed: {test_file}")
            print(f"⏱️  Duration: {duration:.2f}s")
            if result.stderr:
                print("🚨 Error:")
                print(result.stderr)
            if result.stdout:
                print("📤 Output:")
                print(result.stdout)
            return False
            
    except Exception as e:
        print(f"💥 Exception occurred while running tests: {e}")
        return False

def main():
    """Discover and execute all test modules."""
    print("🚀 Starting full test run...")
    
    # Discover test files in this directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = []
    
    for file in os.listdir(test_dir):
        if file.startswith('test_') and file.endswith('.py') and file != 'run_all_tests.py':
            test_files.append(file)
    
    if not test_files:
        print("❌ No tests found.")
        return
    
    print(f"📋 Discovered {len(test_files)} test file(s):")
    for test_file in test_files:
        print(f"  - {test_file}")
    
    # Execute tests
    passed = 0
    failed = 0
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if run_test(test_path):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Summary")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. See log above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
