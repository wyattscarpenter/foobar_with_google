import math

def solution(area, areas=[]):
    if area == 0:
        return areas
    else:
        greatest_sq = math.floor(math.sqrt(area))**2
        return solution(area-greatest_sq, areas + [int(greatest_sq)])

print(solution(12))