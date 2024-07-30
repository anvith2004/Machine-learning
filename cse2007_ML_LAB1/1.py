def sumPairs(lst, target_sum):
    total = 0
    n = len(lst)
    for a in range(n):
        for b in range(a + 1, n):
            if lst[a] + lst[b] == target_sum:
                total += 1
    return total


list_in = input("Enter the list  :")
lst = list(map(int, list_in.split()))
target_sum = int(input("Enter the target sum: "))
answer = sumPairs(lst, target_sum)
print(answer)
