# Example on uses of classes

class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age


    def bark(self):
        print('Woof! My Name is ' + self.name)


    def walk(self, distance):
        print('I just walked a distance of ' + str(distance))

