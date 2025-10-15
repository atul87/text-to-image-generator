#!/usr/bin/env python3
"""
Test script for the image generation improvements.
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the current directory to the path so we can import generate
sys.path.insert(0, str(Path(__file__).parent))

import generate


class TestImageGeneration(unittest.TestCase):
    
    def test_sanitize_filename(self):
        """Test that filename sanitization works correctly."""
        # Test normal text
        self.assertEqual(generate.sanitize_filename("simple text"), "simple text")
        
        # Test text with invalid characters
        self.assertEqual(generate.sanitize_filename("text with / and \\"), "text with  and ")
        
        # Test text with special characters
        self.assertEqual(generate.sanitize_filename("text_with-dashes"), "text_with-dashes")
        
        # Test text that's too long
        long_text = "a" * 100
        self.assertEqual(len(generate.sanitize_filename(long_text)), 50)
        
    def test_get_device(self):
        """Test device selection logic."""
        # Test auto selection with CUDA available
        with patch('torch.cuda.is_available', return_value=True):
            device = generate.get_device("auto")
            self.assertEqual(device.type, "cuda")
            
        # Test auto selection with CUDA not available
        with patch('torch.cuda.is_available', return_value=False):
            device = generate.get_device("auto")
            self.assertEqual(device.type, "cpu")
            
        # Test explicit CPU selection
        device = generate.get_device("cpu")
        self.assertEqual(device.type, "cpu")
        
    def test_load_prompts_from_file(self):
        """Test that prompts can be loaded from file."""
        # Create a temporary test file
        test_file = Path("test_prompts.txt")
        test_content = "prompt 1\n\nprompt 2\n  \n prompt 3\n"
        
        try:
            # Write test content
            with open(test_file, "w") as f:
                f.write(test_content)
            
            # Load prompts
            prompts = generate.load_prompts_from_file(test_file)
            
            # Check results
            self.assertEqual(len(prompts), 3)
            self.assertEqual(prompts[0], "prompt 1")
            self.assertEqual(prompts[1], "prompt 2")
            self.assertEqual(prompts[2], "prompt 3")
            
        finally:
            # Clean up
            if test_file.exists():
                os.remove(test_file)
                
    def test_load_prompts_from_nonexistent_file(self):
        """Test that loading from nonexistent file raises exception."""
        with self.assertRaises(FileNotFoundError):
            generate.load_prompts_from_file("nonexistent_file.txt")


if __name__ == "__main__":
    unittest.main()