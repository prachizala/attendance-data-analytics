"""Unit test suite for punctuality score calculator."""
import unittest

class TestPunctualityCalculator(unittest.TestCase):
    def test_perfect_attendance(self):
        on_time = 20
        total = 20
        score = (on_time / total) * 100
        self.assertEqual(score, 100.0)

if __name__ == '__main__':
    unittest.main()
