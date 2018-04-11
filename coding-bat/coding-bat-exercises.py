def string_match(a, b):
    """Warmup-2 -> string_match:
    Given 2 strings, a and b, return the number of the positions where they contain the same length 2 substring. 
    So "xxcaazz" and "xxbaaz" yields 3, since the "xx", "aa", and "az" substrings appear in the same place in 
    both strings."""

    match = 0
    for i in range(0, len(a) - 1):
        if a[i:i + 2] == b[i:i + 2]:
            match += 1
    return match

def xyz_there(str):
    """String-2 -> xyz_there:
    Return True if the given string contains an appearance of "xyz" where the xyz is not directly preceeded by
    a period (.). So "xxyz" counts but "x.xyz" does not."""

    mark_dot = 0
    mark_nodot = 0
    for i in range(0, len(str) - 1):
        if str[i:i + 4] == ".xyz":
            mark_dot += 1
        elif str[i:i + 3] == "xyz":
            mark_nodot += 1
    return True if mark_nodot - mark_dot > 0 else False

def centered_average(nums):
    """List-2 -> centered-_average:
    Return the "centered" average of an array of ints, which we'll say is the mean average of the values, except 
    ignoring the largest and smallest values in the array. If there are multiple copies of the smallest value, 
    ignore just one copy, and likewise for the largest value. Use int division to produce the final average. 
    You may assume that the array is length 3 or more."""

    nums = sorted(nums)
    nums.pop(0)
    nums.pop(-1)
    return sum(nums) / len(nums)

def end_other(a, b):
    """String-2 -> end_other:
    Given two strings, return True if either of the strings appears at the very end of the other string, ignoring 
    upper/lower case differences (in other words, the computation should not be "case sensitive")."""

    if a.lower()[-len(b.lower()):] == b.lower() or b.lower()[-len(a.lower()):] == a.lower():
        return True
    else:
        return False

def big_diff(nums):
    """List-2 -> big_diff:
    Given an array length 1 or more of ints, return the difference between the largest and 
    smallest values in the array."""

    return max(nums) - min(nums)

def string_bits(str):
    """Warmup-2 -> string_bits:
    Given a string, return a new string made of every other char starting with the first, so "Hello" yields "Hlo"."""
    new_str = ""
    for i in range(0, len(str), 2):
        new_str += str[i]
    return new_str

def string_times(str, n):
    """Warmup-2 -> string_times:
    Given a string and a non-negative int n, return a larger string that is n copies of the original string."""

    return str * n

def count_code(str):
    """String-2 -> count_code:
    Return the number of times that the string "code" appears anywhere in the given string, except we'll accept any 
    letter for the 'd', so "cope" and "cooe" count."""

    count = 0
    for i in range(0, len(str) - 1):
        try:
            if str[i:i + 2] == "co" and str[i + 3] == "e":
                count += 1
        except:
            pass
    return count

def count_evens(nums):
    """List-2 -> count_evens:
    Return the number of even ints in the given array."""

    even = 0
    for num in nums:
        if num % 2 == 0:
            even += 1
    return even

def string_splosion(str):
    """Warmup-2 -> string-splosion:
    Given a non-empty string like "Code" return a string like "CCoCodCode"."""

    new_string = ""
    for i in range(0, len(str)):
        new_string += str[0:i + 1]
    return new_string


if __name__ == "__main__":
    """Testing code snippets"""
    print("1. string_match:", {True: True, False: False}[string_match('xxcaazz', 'xxbaaz') == 3])
    print("2. xyz_there:", {True: True, False: False}[xyz_there('abcxyz') == True])
    print("3. centered_average:", {True: True, False: False}[centered_average([1, 2, 3, 4, 100]) == 3])
    print("4. end_other:", {True: True, False: False}[end_other('Hiabc', 'abc') == True])
    print("5. big_diff:", {True: True, False: False}[big_diff([10, 3, 5, 6]) == 7])
    print("6. string_bits:", {True: True, False: False}[string_bits('Hello') == 'Hlo'])
    print("7. string_times:", {True: True, False: False}[string_times('Hi', 2) == 'HiHi'])
    print("8. count_code:", {True: True, False: False}[count_code('aaacodebbb') == 1])
    print("9. count_even:", {True: True, False: False}[count_evens([2, 1, 2, 3, 4]) == 3])
    print("10. string_splosion:", {True: True, False: False}[string_splosion('Code') == 'CCoCodCode'])




