import unittest

from evoltier.test import TestProbabilityDistribution
from evoltier.test import TestMultiVariableGaussian


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.makeSuite(TestProbabilityDistribution))
    test_suite.addTests(unittest.makeSuite(TestMultiVariableGaussian))
    return test_suite
