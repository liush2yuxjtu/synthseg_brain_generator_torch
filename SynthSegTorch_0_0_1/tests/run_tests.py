"""Test runner for SynthSegTorch

This script runs all the tests for the SynthSegTorch package.
Tests use try-except blocks instead of unittest or pytest.
"""

import os
import sys
import importlib
import time

# Add parent directory to path to allow importing SynthSegTorch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def discover_tests():
    """Discover all test files in the tests directory."""
    test_files = []
    for file in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        if file.startswith('test_') and file.endswith('.py'):
            test_files.append(file[:-3])  # Remove .py extension
    return test_files


def run_all_tests():
    """Run all discovered tests and report results."""
    test_files = discover_tests()
    test_results = {}
    
    print(f"Discovered {len(test_files)} test files")
    print("Running tests...\n")
    
    start_time = time.time()
    
    for test_file in test_files:
        try:
            # Import the test module
            test_module = importlib.import_module(f"tests.{test_file}")
            
            # Find all test functions in the module
            test_functions = [func for func in dir(test_module) 
                             if func.startswith('test_') and callable(getattr(test_module, func))]
            
            # Run each test function
            for test_func_name in test_functions:
                test_func = getattr(test_module, test_func_name)
                print(f"Running {test_file}.{test_func_name}...")
                try:
                    result = test_func()
                    test_results[f"{test_file}.{test_func_name}"] = result
                except Exception as e:
                    print(f"Error running {test_file}.{test_func_name}: {str(e)}")
                    test_results[f"{test_file}.{test_func_name}"] = False
        except Exception as e:
            print(f"Error importing {test_file}: {str(e)}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print summary
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    print(f"\nTest Summary: {passed}/{total} tests passed ({elapsed_time:.2f} seconds)")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed:")
        for test_name, result in test_results.items():
            if not result:
                print(f"  - {test_name}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)