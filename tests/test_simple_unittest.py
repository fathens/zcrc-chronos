"""
単純なunittestモジュール
"""

import unittest


class SimpleTest(unittest.TestCase):
    """
    単純なテストケース
    """

    def test_simple(self):
        """
        単純なテスト関数
        """
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
