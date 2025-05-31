# TDD Plan for Converting SynthSeg from TensorFlow/Keras to PyTorch

## Overview

This document outlines the Test-Driven Development (TDD) approach for converting the SynthSeg brain generator from TensorFlow/Keras to PyTorch. The conversion will follow these main steps:

1. Create a new directory structure for the PyTorch implementation
2. Rename TensorFlow/Keras files, functions, and classes with `_tf` postfix
3. Create PyTorch equivalents of these components
4. Implement test cases using try-except blocks to validate the conversion
5. Ensure functional equivalence between the original and converted implementations

## Directory Structure

```
SynthSegTorch_0_0_1/
├── __init__.py
├── ext/
│   ├── __init__.py
│   ├── lab2im/
│   │   ├── __init__.py
│   │   ├── edit_tensors.py
│   │   ├── edit_tensors_tf.py
│   │   ├── edit_volumes.py
│   │   ├── image_generator.py
│   │   ├── image_generator_tf.py
│   │   ├── lab2im_model.py
│   │   ├── lab2im_model_tf.py
│   │   ├── layers.py
│   │   ├── layers_tf.py
│   │   ├── utils.py
│   ├── neuron/
│       ├── __init__.py
│       ├── layers.py
│       ├── layers_tf.py
│       ├── models.py
│       ├── models_tf.py
│       ├── utils.py
│       └── utils_tf.py
├── SynthSeg/
│   ├── __init__.py
│   ├── brain_generator.py
│   ├── brain_generator_tf.py
│   ├── labels_to_image_model.py
│   ├── labels_to_image_model_tf.py
│   ├── metrics_model.py
│   ├── metrics_model_tf.py
│   ├── model_inputs.py
│   ├── model_inputs_tf.py
│   ├── training.py
│   └── training_tf.py
├── tests/
│   ├── __init__.py
│   ├── test_brain_generator.py
│   ├── test_edit_tensors.py
│   ├── test_image_generator.py
│   ├── test_lab2im_model.py
│   ├── test_labels_to_image_model.py
│   ├── test_layers.py
│   ├── test_metrics_model.py
│   ├── test_model_inputs.py
│   ├── test_neuron_layers.py
│   ├── test_neuron_models.py
│   ├── test_neuron_utils.py
│   ├── test_training.py
│   └── test_utils.py
└── requirements.txt
```

## Conversion Strategy

### Phase 1: Setup and Preparation

1. Create the directory structure for SynthSegTorch_0_0_1
2. Copy all files from the original SynthSeg repository
3. Create a requirements.txt file with PyTorch dependencies

### Phase 2: Rename TensorFlow/Keras Components

1. For each file containing TensorFlow/Keras code:
   - Rename the file to include `_tf` postfix
   - Create an empty file with the original name for the PyTorch implementation

2. For each function/class using TensorFlow/Keras:
   - Rename the function/class to include `_tf` postfix in the `_tf` files
   - Update all references to these functions/classes throughout the codebase

### Phase 3: Test Case Development

1. Create test files for each component to be converted
2. Implement test cases using try-except blocks instead of unittest/pytest
3. Each test should:
   - Load the same input data for both implementations
   - Process the data through both the TensorFlow and PyTorch implementations
   - Compare the outputs to ensure they are functionally equivalent
   - Handle exceptions gracefully with informative error messages

### Phase 4: PyTorch Implementation

1. Implement PyTorch equivalents for each TensorFlow/Keras component:
   - Convert Keras layers to PyTorch modules
   - Replace TensorFlow operations with PyTorch equivalents
   - Ensure tensor dimension ordering is consistent
   - Adapt model building and training code to PyTorch paradigms

2. Key components to convert:
   - Custom layers (RandomSpatialDeformation, SampleConditionalGMM, etc.)
   - Model architectures (UNet, etc.)
   - Loss functions and metrics
   - Training loops and optimization

## Testing Approach

### Test Case Structure

Each test file will follow this general structure:

```python
def test_component_name():
    try:
        # Setup test data
        test_input = create_test_input()
        
        # Run TensorFlow implementation
        import tensorflow as tf
        from SynthSegTorch_0_0_1.component_name_tf import function_name_tf
        tf_output = function_name_tf(test_input)
        
        # Run PyTorch implementation
        import torch
        from SynthSegTorch_0_0_1.component_name import function_name
        torch_output = function_name(test_input)
        
        # Convert outputs to numpy for comparison
        tf_output_np = tf_output.numpy() if hasattr(tf_output, 'numpy') else tf_output
        torch_output_np = torch_output.detach().cpu().numpy() if hasattr(torch_output, 'detach') else torch_output
        
        # Compare outputs
        assert np.allclose(tf_output_np, torch_output_np, rtol=1e-5, atol=1e-5), \
            f"Outputs don't match: TF={tf_output_np.mean()}, Torch={torch_output_np.mean()}"
        
        print(f"✓ {function_name} test passed!")
        return True
    except Exception as e:
        print(f"✗ {function_name} test failed: {str(e)}")
        return False
```

### Test Runner

A main test runner will execute all tests and report results:

```python
def run_all_tests():
    test_results = {
        "test_brain_generator": test_brain_generator(),
        "test_edit_tensors": test_edit_tensors(),
        # ... other tests
    }
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    print(f"\nTest Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed:")
        for test_name, result in test_results.items():
            if not result:
                print(f"  - {test_name}")

if __name__ == "__main__":
    run_all_tests()
```

## Key Conversion Challenges

1. **Tensor Dimension Ordering**: TensorFlow uses channel-last format (NHWC) while PyTorch uses channel-first (NCHW)

2. **Layer Implementation Differences**: 
   - TensorFlow/Keras layers maintain state differently than PyTorch modules
   - Custom layer implementation patterns differ significantly

3. **Random Operations**: 
   - Ensuring consistent random number generation between frameworks
   - Matching random transformations and augmentations

4. **Graph vs Eager Execution**: 
   - TensorFlow's graph-based execution vs PyTorch's eager execution
   - Handling static vs dynamic computation graphs

5. **Loss Functions and Metrics**: 
   - Different implementations and reduction strategies
   - Ensuring numerical stability in both frameworks

## Implementation Priority

1. Core utility functions (edit_tensors, utils)
2. Basic layers (SpatialTransformer, etc.)
3. Model building blocks (lab2im_model)
4. High-level components (brain_generator, labels_to_image_model)
5. Training and evaluation code

## Validation Strategy

For each component:

1. Unit tests to verify individual functions and classes
2. Integration tests to verify component interactions
3. End-to-end tests to verify the complete pipeline
4. Visual inspection of generated images to ensure quality

## Timeline

1. **Week 1**: Setup, preparation, and TensorFlow component renaming
2. **Week 2**: Core utility functions and basic layers conversion
3. **Week 3**: Model building blocks conversion
4. **Week 4**: High-level components conversion
5. **Week 5**: Training and evaluation code conversion
6. **Week 6**: Testing, debugging, and documentation

## Conclusion

This TDD plan provides a structured approach to converting the SynthSeg brain generator from TensorFlow/Keras to PyTorch. By following this plan, we can ensure that the converted implementation maintains functional equivalence with the original while leveraging the benefits of PyTorch's dynamic computation graph and Python-native design.