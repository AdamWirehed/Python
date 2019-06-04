# A dictionary is similar to a list. They are both collections
# of items but a dictionary has key and values (opposed to just
# values in a list). You have some "key" (a word or number) and
# you "lookup" that word in your dictionary to get the word's
# definition or "value"

# Documentation: Decent

# Defining an empty dictionary
dicc = {}

# Adding stuff to dictionary
dicc['Fish'] = 'Animal that neither Alex or Gabe eats. Tends to live in water.'
dicc['Apple'] = 'Spherious red or green object that seems eatable... Need to investigate that more.'
dicc['Elon Musk'] = 'As close as we get to Batman in the engineering industry.'
dicc[3] = 'What did you really expect?'
dicc['A list'] = ['Oh', 'boy', 'A list in a dictionary!']
dicc['Inception'] = {'Nested': 'When one dictionary is a VALUE in a dictionary.'}

# Print out the dictionary
print(str(dicc))
print()

print('Looping...')
print()
# Loop over all keys in the dictionary
for key in dicc:
    # Calling str due to "3"
    print('Key: ' + str(key))
    # Print the value by using the key to "look it up"
    print('Value: ' + str(dicc[key]))
    print()
