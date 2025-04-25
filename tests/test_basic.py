import unittest
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestBasicFunctionality(unittest.TestCase):
    """Basic test cases."""

    def test_config_template_exists(self):
        """Test that config_template.py exists."""
        self.assertTrue(os.path.exists('config_template.py'))
    
    def test_readme_exists(self):
        """Test that README.md exists."""
        self.assertTrue(os.path.exists('README.md'))
    
    def test_gitignore_exists(self):
        """Test that .gitignore exists."""
        self.assertTrue(os.path.exists('.gitignore'))

if __name__ == '__main__':
    unittest.main()