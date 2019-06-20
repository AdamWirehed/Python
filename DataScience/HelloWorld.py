# Simply testing libs and the env for python3 in Atom

import numpy as np
import matplotlib.pyplot as plt

for i in range(1, 5):
    print('Hello World')

x = np.random.rand(5, 6)
y = np.random.rand(5, 6)

# plt.scatter(x, y)
# plt.show()

(age, income) = '32,60.000'.split(',')
print('Age = {} years'.format(age))
print('Income = {} kr/months \n'.format(income))

personAge = dict()
personAge['Adam'] = 21
personAge['Julia'] = 17
personAge['Zäther'] = 21
personAge['Fabbe'] = 22

for person in personAge:
    print(('Name: ' + person + ', age {}').format(personAge.get(person)))
print('\n')

wordList = ['ape', 'fish', 'lobster', 'doorknob', 'fish', 'yellow', 'indigo',
            'ape', 'lobster']
print(wordList)
uniqueWords = set()

for word in wordList:
    print(word)
    uniqueWords.update([word])

print(uniqueWords)
