# Intcode program from "Advent of Code: Day 2 and Day 5"


def opcode(act, ix, arglist):

    # Getting the values depending on the instruction
    if act > 2:
        i1 = arglist[ix + 1]
        k = 2
    else:
        i1 = arglist[ix + 1]
        i2 = arglist[ix + 2]
        i3 = arglist[ix + 3]
        k = 4

    # The different operations
    def first():
        arglist[i3] = arglist[i1] + arglist[i2]

    def second():
        arglist[i3] = arglist[i1] * arglist[i2]

    def third():
        iv = int(input("Enter value : "))
        arglist[i1] = iv

    def fourth():
        print(arglist[i1])

    def default():
        print("Unvalid opcode")

    dictmap = {
        1: first,
        2: second,
        3: third,
        4: fourth,
    }

    dictmap.get(act, default)()
    return k


# ------------------- main function ------------------------

testList = [3,0,4,0,99]
print(testList)

# Index counter
i = 0
jump = 0

while i < len(testList):
    arg = testList[i]

    if arg == 99:
        break

    jump = opcode(arg, i, testList)
    i += jump

print(testList)
