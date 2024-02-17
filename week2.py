from spicy import stats
from scipy.stats import t

# exercise 1
# a. chi square test

# b. H0: the count in our sample is evenly distributed
#    H1: the count in our sample is not evenly distributed

# c. x**2 = ?
scissors = 30
paper = 24
rock = 21
# x = ((21 - 25) ** 2 / 25) + ((24 - 25) ** 2 / 25) + ((30 - 25) ** 2 / 25)
# 25 because we evenly distribute the values (we divide 75 by 3)
# print(x)

# d. sample would be bigger than x**2
# m- 1 or df are the number of items (rock, paper, scissors) minus 1
# print(1 - stats.chi2.cdf(x, 3 - 1))

# e. p-value is bigger than x**2 so we can't reject H0

# f. based on the results, we can't say they have a preference

# exercise 2
# a. chi square test

# b. H0: the hair color count in our sample may not differ from the expected distribution
#    H1: the hair color count in our sample may differ from the expected distribution

# c. x**2 = ?
expected_red = 19
expected_black = 3
y = expected_black + expected_red
expected_brown = 30 / 100 * (246 - y)
expected_blonde = 30 / 100 * (246 - y)
expected_dark = 25 / 100 * (246 - y)

x = ((83 - expected_brown) ** 2 / expected_brown) + ((76 - expected_blonde) ** 2 / expected_blonde) + (
        (65 - expected_dark) ** 2 / expected_dark)
# print(x)

# d. p-value
# print(1 - stats.chi2.cdf(x, 3 - 1))

# e. with 95% reliability, can H0 be rejected or not? yes

# f. no

# exercise 3
average = 975
standard_deviation = 100
sample_size = 30
confidence_level = 0.95

# a. limit to be 95% sure

# # Calculate the standard error
standard_error = standard_deviation / (sample_size ** 0.5)

# Calculate the margin of error
margin_of_error = stats.t.ppf((1 + confidence_level) / 2, sample_size - 1) * standard_error

# Calculate the lower and upper limits of the confidence interval
lower_limit = average - margin_of_error
upper_limit = average + margin_of_error

# print(lower_limit, upper_limit)

# b. sample_size = 100
standard_error = standard_deviation / (100 ** 0.5)
margin_of_error = stats.t.ppf((1 + confidence_level) / 2, 100 - 1) * standard_error
lower_limit1 = average - margin_of_error
upper_limit1 = average + margin_of_error

# print(lower_limit1, upper_limit1)

# exercise 4
alpha = 0.01
confidence_level = 1 - alpha
sample_size = 100
degrees_of_freedom = sample_size - 1
sample_mean = 11.9  # Sample mean
sample_std = 1  # Sample standard deviation

# a. what is the factor (t-value)?
t = t.ppf(confidence_level, degrees_of_freedom)
# print(t)

# b. calculate acceptance area
margin_of_error = t * (sample_std / (sample_size ** 0.5))
lower_bound = sample_mean - margin_of_error
upper_bound = sample_mean + margin_of_error

print(lower_bound, upper_bound)
# do we need to adjust the machine? no need to adjust the machine because 11.9 = x is included in [11.74, 12.16]
