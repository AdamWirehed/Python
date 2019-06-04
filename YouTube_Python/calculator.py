def add(num1, num2):
    """Returns num1 plus num2"""
    return num1 + num2

def sub(num1, num2):
    """Returns num1 minus num2"""
    return num1 - num2

def mul(num1, num2):
    """Returns num1 multiplied with num2"""
    return num1 * num2

def div(num1, num2):
    """Returns num1 divided with num2"""
    # Tests if the variable num2 is assigned zero
    try:
        return num1 / num2
    except ZeroDivisionError:
        return 'Divide something with 0? Do you want this laptop to freeze?'

def runOperation(operation, num1, num2):
    """Determines the operation to run based on the operation argument
    which should be passed in as an int"""
    # Determine operation
    if operation == 1 or operation == '+':
        print('Adding... ' + str(num1) + ' with ' + str(num2))
        print(add(num1, num2))
    elif operation == 2 or operation == '-':
        print('Subtracting... ' + str(num1) + ' with ' + str(num2))
        print(sub(num1, num2))
    elif operation == 3 or operation == '*':
        print('Multiplying... ' + str(num1) + ' with ' + str(num2))
        print(mul(num1, num2))
    elif operation == 4 or operation == '/':
        print('Dividing... ' + str(num1) + ' with ' + str(num2))
        print(div(num1, num2))
    else:
        print('Qué? No comprendo!')

def calc():
    # Starting loop
    hell = True
    while hell:
        validinput = False
        while not validinput:
            # Get user input
            try:
                # If the input is valid, go through
                num1 = int(input('What is number 1? '))
                num2 = int(input('What is number 2? '))
                operation = int(input('What do you want to do? \n1) add \n2) sub \n3) mul \n4) div \n'
                                    'Enter number here: '))
                validinput = True
            except ValueError:
                # If the input is not valid, give the user a new opportunity
                print('Invalid input. Try again.')
            except:
                # Always use "except:" last, otherwise it will trigger before more specific exceptions
                print('Unknown error')
            runOperation(operation, num1, num2)

        loop = input('Do you want to stop? yes or anything: ')
        if loop == 'yes':
            # Break the loop
            hell = False
        else:
            # Keep the loop
            hell = True
        print('Rebooting...\n')

# Determining if the "calc" function should be run
# if __name__ == '__calc__':
calc()