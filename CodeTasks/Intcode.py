# Intcode program from "Advent of Code: Day 2 and Day 5"


def opcode(act, ix, arglist):
    actstr = str(act)
    opc = int(actstr[-2:])  # Gets the Opcode
    cix = len(actstr) - 2
    im = [0] * cix        # Empty list to hold the values for the operations

    # Loop for checking immediate mode
    while cix <= 0:
        if actstr[cix] == 0:
            im[cix] = arglist[ix]
        else:
            im[cix] = int(actstr[cix])

        cix -= 1

    def first():
        arglist[im[3]] = im[1] + im[2]

    def second():
        arglist[im[3]] = im[1] * im[2]

    def third():
        iv = int(input("Enter value : "))
        arglist[i1] = iv

    def fourth():
        print(im[1])

    def default():
        print("Unvalid opcode")

    dictMap = {
        1: first,
        2: second,
        3: third,

    }

    dictMap.get(opc, default)()


# ------------------- main function ------------------------

testList = [3,0,4,0,99]
print(testList)

# Index counter
i = 0

while i < len(testList):
    arg = testList[i]

    if arg == 99:
        break

    opcode(arg, i, testList)
    print(testList)
