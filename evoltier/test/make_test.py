import unittest


def suite():
    test_suite = unittest.TestSuite()
    all_test_suite = unittest.defaultTestLoader.discover(".", pattern="test_*.py")
    for test in all_test_suite:
        test_suite.addTest(test)
    return test_suite

if __name__ == "__main__":
    suites = suite()
    unittest.TextTestRunner().run(suites)