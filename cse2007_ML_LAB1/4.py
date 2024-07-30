def max_Char(in_st):
    char_count = {}
    for char in in_st:
        if char.isalpha():
            char = char.lower()
            if char in char_count:
                char_count[char] += 1
            else:
                char_count[char] = 1
    if not char_count:
        return "no alphabets"
    max_char = None
    max_count = 0
    for char, count in char_count.items():
        if count > max_count:
            max_char = char
            max_count = count

    return max_char, max_count

in_st = input("Enter a string: ")
max_char, max_count = max_Char(in_st)
if max_char == "no alphabets":
    print(max_char)
else:
    print(f"Max character: '{max_char}' with count: {max_count}")
