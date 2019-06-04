#!/bin/python
# This is a program called fizzbuzz. It is often used to test basic programming capabilities.
# The specification is as follows:

# For numbers 1 through 100,

#   if the number is divisible by 3 print Fizz;
#   if the number is divisible by 5 print Buzz;
#   if the number is divisible by 3 and 5 (15) print FizzBuzz;
#   else, print the number.
#
d1=3
d2=5
d3=15

for i in range(100):
    if i % d1 == 0 and i % d2 != 0:
        print("Fizz")
    elif i % d2 == 0 and i % d1 != 0:
        print("Buzz")
    elif i % d3 == 0:
        print("FizzBuzz")
    else:
        print("Skiten, %d, går inte att dela med %d eller %d" % (i,d1,d2))
