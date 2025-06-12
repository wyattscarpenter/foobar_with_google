def solution(l): #I'm sure there's a very functional way to do this that's basically exactly the same but would have saved me a few lines.
  count = 0
  #s = set()
  #It's kind of ambiguous if this should be, like, a set, and multiples of the same triple should be thrown out, or what. I mean, the question seems to be phrased to imply you keep the mulitples.
  #...Changing from one to the other apparently makes no difference; the open tests pass the and hidden ones fail, still...
  #time.sleep(100) #OK... sleep causes test fails so one of these ways of thinking about it is right, but I'm timing out too fast to see it...
  for i,x in enumerate(l): #This solution wastes compute but that's not very important rn
    for j,y in enumerate(l[i+1:]):
      for k,z in enumerate(l[j+1:]):
        #if i<j and j<k and y%x==0 and z%y==0: s.add((x,y,z))
        count += 1 if i<j and j<k and y%x==0 and z%y==0 else 0 #maybe this is branchless and thus fast? Ah, well... "python", "fast".
  #return len(s)
  return count

def solution(l):
  count = 0
  ll = len(l)
  for i in range(ll): #This solution wastes compute but that's not very important rn
    for j in range(i+1, ll):
      for k in range(j+1, ll):
        #if i<j and j<k and y%x==0 and z%y==0: s.add((x,y,z))
        count += 1 if l[j]%l[i]==0 and l[k]%l[j]==0 else 0 #maybe this is branchless and thus fast? Ah, well... "python", "fast".
  #return len(s)
  return count

#This did not even really work:
import functools
@functools.cache
def what_in_this_list_divides_this_number_and_first_of_all_where_are_they(l,n):
  pairs = []
  for index, value in enumerate(l):
    if n%value==0: pairs += [(index,value)]
  #print(pairs)
  return pairs
def solution(l): #OK, the previous solutions were too slow, so I need to use... the beauty of algorithms!
# The plan: instead of doing it O(n^3) style, make a list of all the divisible pairs and construct from those.
# Maybe blast some functools instead?
  l = tuple(l)
  count = 0
  ll = len(l)
  for i in range(ll):
    j_tuples = [p for p in what_in_this_list_divides_this_number_and_first_of_all_where_are_they(l,l[i]) if p[0]>i]
    for j in j_tuples:
      k_tuples = [new for new in what_in_this_list_divides_this_number_and_first_of_all_where_are_they(l,j[1]) if new[0]>j[0]]
      count += len(k_tuples)
  return count

#OK, this passes everything but hidden test 5
from collections import defaultdict
def solution(l):
  count=0
  divisor_map = defaultdict(set) # dict int -> (set(int)) (the ints being the indices of the list items that can divide that number.)
  ll = len(l)
  r = list(reversed(l)) #for some reason it doesn't iterate over this in the for loop unless you coerce to list? Weird. Maybe a python bug?
  for index, value in enumerate(r):
    #print("index:", index)
    for inner_index, inner_value in enumerate(r):
      if inner_index>index and (value%inner_value)==0:
        #print(index, value, inner_index, inner_value)
        divisor_map[index].add(inner_index)
  #print(divisor_map)
  for value in divisor_map:
    #print(value)
    for second_index in divisor_map[value]:
      if second_index in divisor_map:
        count += len(divisor_map[second_index]) #this used to be a nested for loop to be extra procedural or whatever but boy howdy was that a speed-mistake!
  #print("count", count)
  return count

# 3 1 4 0 1 1 4
print(solution([1, 2, 3, 4, 5, 6]), solution([1, 1, 1]), solution([1, 1, 1, 1]), solution([2, 3, 7, 19]), solution([2, 4, 8]), solution([2, 3, 9, 81]), solution([1, 2, 4, 3, 8]))
print(solution([1]*2000))
"""
-- Python cases -- 
Input:
solution.solution([1, 2, 3, 4, 5, 6])
Output:
    3

Input:
solution.solution([1, 1, 1])
Output:
    1
"""