import math

class Solution:
    def isPalindrome(self, x: int) -> bool:
        list_nr = [i for i in str(x)]
        rev_list_nr = list_nr[::-1]
        for ix in range(math.ceil(len(list_nr)/2)):
            if list_nr[ix] != rev_list_nr[ix]:
                return False
        
        return True

if __name__ == "__main__":
    sol = Solution()
    print(sol.isPalindrome(12321))