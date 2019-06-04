# Use imported .py files, for loops and lists
# Documentation: Good

# Importing "time", "random" and homemade "calculator"
import time
import random
import calculator

# Loop a random number of times (Between 10 and 50)
loopMax = random.randint(10, 50)

# Print "mathing"
print('Request for ' + str(loopMax) + ' maths received... Beginning mathing sequence...')

# Define python list of operation
# This list represents the strings for the 4 calculator operations
# Create the list just with addition in it and then add to it, [] for empty list
operations = ['+']
# Using append (A function that is integrated in python) to add to the end of the list
operations.append('-')
operations.append('/')
operations.append('*')

# Print the number of operations that will be used
# Using "len()" function to get return the integer count
# of the number of items in the list
print('Operation count ' + str(len(operations)))
# Showing what the list contains
print('This laptop will maths using following operations ' + str(operations))
print('\n\n')

for index in range(loopMax):
    # Getting random numbers to operate
    num1 = random.randint(-99, 99)
    num2 = random.randint(-99, 99)

    print('Matching with ' + str(num1) + ' and ' + str(num2))

    # Generate random sleep time between 0 and 2 seconds
    sleepTime = random.randint(0, 1)

    print('IN FOR LOOP:')
    # Looping in the "operations" list and assigning variable "operation" with an string from the list
    for operation in operations:
        # Print the operation
        print('operation = ' + operation)
        # Using calculator.py to execute the operation depending on the variable
        calculator.runOperation(operation, num1, num2)
        print() # Just to make a new line

    print('\n')
    print('INDEX FOR LOOP')
    """Loop using an index counter"""
    for i in range(len(operations)):
        print('i = ' + str(i))
        print('operation = ' + operations[i])
        calculator.runOperation(operations[i], num1, num2)
        print()

    # Sleep for random time
    time.sleep(sleepTime)
    print('\n-------------------\n')



