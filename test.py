from itertools import combinations

l = [('a', 'b', 'c'), ('d', 'e', 'f'), ('g', 'e', 'f'), ('h', 'b', 'c')]
print([(a[0], b[0]) for a, b in combinations(l, 2) if a[1:] == b[1:]])
