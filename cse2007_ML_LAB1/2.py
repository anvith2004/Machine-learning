def cal_range(lst):
    if len(lst) < 3:
        return "not possible"
    
    min_number = min(lst)
    max_number = max(lst)
    range_value = max_number - min_number
    return range_value

list_in = input("Enter the list :")
lst = list(map(int, list_in.split()))
result = cal_range(lst)
print("range is:", result)
