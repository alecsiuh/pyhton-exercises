# ---------- PROBABILITIES ----------
# exercise 1
# roll 2 dice. chance you get a 5 and a 2

five = 1 / 6
two = 1 / 6

P_five_AND_two_OR_two_AND_five = (five * two) + (two * five)
# print(round(P_five_AND_two_OR_two_AND_five, 4))

# exercise 2
# 2 dice. one 3 red areas and 3 yellow areas and one 5 red areas and 1 yellow area
# both red and yellow area

P_red_AND_yellow_OR_yellow_AND_red = (3 / 6 * 1 / 6) + (3 / 6 * 5 / 6)
# print(P_red_AND_yellow_OR_yellow_AND_red)

# exercise 3
yellow = 4
brown = 2
orange = 1
green = 1
purple = 2
marbles = yellow + brown + orange + green + purple

# a. 4 pulled marbles are all yellow
P_all_yellow = yellow / marbles * (yellow - 1) / (marbles - 1) * (yellow - 2) / (marbles - 2) * (yellow - 3) / (
        marbles - 3)
# print(round(P_all_yellow, 3))

# b. pull 2 marbles. brown and green
P_brown_AND_green_OR_green_AND_brown = (brown / marbles * green / (marbles - 1)) + (
        green / marbles * brown / (marbles - 1))
# print(round(P_brown_AND_green_OR_green_AND_brown, 3))

# c. one marble. orange or green
P_orange_OR_green = 1 / marbles + 1 / marbles
# print(round(P_orange_OR_green, 3))

# exercise 4
students = 144
basic = 90
advanced = 60
basic_AND_advanced = 30

# chance that one student passed one of the two exams
P_basic_OR_advanced = basic / students + advanced / students - basic_AND_advanced / students
# print(round(P_basic_OR_advanced, 3))

# exercise 5
students_1 = 1000
c = 400
b = 200
c_AND_b = 100

# change of a random student playing one of the two
P_c_OR_b = c / students_1 + b / students_1 - c_AND_b / students_1
# print(round(P_c_OR_b, 3))

# exercise 6
passed = 0.7
# P(day | pass)
day = 0.5
# P(evening | pass)
evening = 0.5
participating_day = 0.6
participating_evening = 0.4

# a. successful student
successful_student = passed

# c. successful student, given he is a day student
P_successful_given_day = day

# b. successful day student
P_passed_AND_day = passed * day

# d. P(passed | day)
P_passed_given_day = (passed * day) / participating_day

# e. P(failed | day)
P_failed_given_day = 1 - P_passed_given_day

# f. failed day student
P_failed_AND_day = day * P_failed_given_day

# exercise 7
coats = 3
# chance they get their own coat
P_own_coat = 1 / coats * 1 / (coats - 1) * 1 / (coats - 2)

# exercise 8
# P(flooded | hard rain)
flooded_given_hard_rain = 0.5
rain = 35 / 365 * 100
flooded = 20 / 365 * 100

# P(hard rain | flooded)
P_hard_rain_given_flooded = (flooded_given_hard_rain * flooded) / rain
# print(round(P_hard_rain_given_flooded, 4))

# exercise 9
windows = 0.9
# P(windows | crashed)
P_windows_given_crashed = 0.999
crashed = 0.01

# P(crashed | windows)
P_crashed_given_windows = (P_windows_given_crashed * crashed) / windows
# print(round(P_crashed_given_windows, 4))

# exercise 10
# P(positive | infected)
positive_given_infected = 0.999
# P(negative | not infected)
negative_given_not_infected = 0.99
carrier = 0.6

# P(infected | positive)

from spicy import stats

# ---------- PROBABILITY DISTRIBUTION ----------
# exercise 1
# roll dice 4 times

# a. get exactly one 6
# # number of times you do the experiment
# n = 4
# # how many times can the value be produced?
# p = 1 / 6
# # how many times do you want the value to be produced
# x = 1
# print(round(stats.binom.pmf(x, n, p), 4))

# four sixes
# n = 4
# p = 1/6
# x = 4
# print(round(stats.binom.pmf(x, n, p), 4))

# 2 throws under the 3 and 2 throws 3 or higher
# n = 4
# p = 2/6
# x = 2
# print(round(stats.binom.pmf(x, n, p), 4))

# exercise 2
# exactly 4 passed
# n = 6
# p = 0.75
# x = 4
# print(round(stats.binom.pmf(x, n, p), 4))

# exactly 5 passed
# n = 6
# p = 0.75
# x = 5
# print(round(stats.binom.pmf(x, n, p), 4))

# exactly 6 passed
# n = 6
# p = 0.75
# x = 6
# print(round(stats.binom.pmf(x, n, p), 4))

# less than 4
# n = 6
# p = 0.75
# x = range(4)
# print(stats.binom.cdf(x, n, p))
# # get the last number

# exercise 3
# traffic_light = 6
# red = 0.4

# no red lights
# n = traffic_light
# p = red
# x = 0
# print(round(stats.binom.pmf(x, n, p), 4))

# all red lights
# n = traffic_light
# p = red
# x = 6
# print(round(stats.binom.pmf(x, n, p), 4))

# at least 2 red lights
# print(1 - stats.binom.cdf(2, 6, 0.4))

# time lost
# average_red_lights = red * traffic_light
# time_lost = 2 * average_red_lights

# exercise 4
# 2 flat tires a year

# a. no flat tire
# print(stats.poisson.pmf(0, 2))

# b. more than 3
# print(1 - stats.poisson.cdf(3, 2))

# c. 2 flat tires in 1 month
# prob_per_month = 2/12
# print(stats.poisson.pmf(2, prob_per_month))

# exercise 5
# average_emails_per_day = 20
# emails_in_a_specific_day = 100

# a. more than 100
# print(1 - stats.poisson.cdf(100, 20))

# b. more than 30
# print(1 - stats.poisson.cdf(30, 20))

# c. exactly 20
# print(stats.poisson.pmf(20, 20))

# d. 10 or less
# print(stats.poisson.cdf(10, 20))

# e. 650 or less in a month
# average_a_month = 20 * 30
# print(stats.poisson.cdf(650, average_a_month))

# exercise 6
# average_transactions_per_second = 3.5
# maximum of transactions_per_second = 7

# a. 7 or more
# print(1 - stats.poisson.cdf(7, 3.5))

# b. no transaction
# print(round(stats.poisson.cdf(0, 3.5), 4))

# c. 3 or fewer
# print(stats.poisson.cdf(3, 3.5))

# d. 2 or more
# print(1 - stats.poisson.cdf(2, 3.5))

# e. per day
# print((3.5*60)*60*24)

# exercise 7
# questions = 40
# average_difficulty = 0.85
# points_per_exercise = 1

# expected_value = 40 * 0.85
# standard_deviation = math.sqrt(40 * 0.85 * (1 - 0.85))
# print(expected_value)
# print(round(standard_deviation), 2)

# the answer is a
# expected_value = 20
# standard_deviation = 4

# a. hair longer than 28cm
# print(stats.norm.cdf(expected_value, 28, standard_deviation))

# b. shorter than 16cm
# print(1 - stats.norm.cdf(expected_value, 16, standard_deviation))

# c. between 18cm and 22cm
# print(stats.norm.cdf(expected_value, 18, standard_deviation) - stats.norm.cdf(expected_value, 22, standard_deviation))

# exercise 10
# expected_value = 50
# standard_deviation = 5

# a. lower than 40
# print(1 - stats.norm.cdf(expected_value, 40, standard_deviation))

# b. between 42 and 52
# print(stats.norm.cdf(expected_value, 42, standard_deviation) - stats.norm.cdf(expected_value, 52, standard_deviation))

# c. higher than 58.75
# print(stats.norm.cdf(expected_value, 58.75, standard_deviation))
# print (4/100 * 75)
