# import abstract class
from evoltier.model import probability_distribution
ProbabilityDistribution = probability_distribution.ProbabilityDistribution

# import concrete class
# HACK: The 'from' statement is not compliant to PEP8.
from evoltier.model import multi_variable_gassian
MultiVariableGaussian = multi_variable_gassian.MultiVariableGaussian
