def to_base(i,b,s=""):
  while i:
    s=str(i%b)+s
    i=i//b
  return s

def based_subtraction(x,y,base):
  result = int(x, base) - int(y, base)
  return to_base(result, base)

def solution(n, base):
  """Given a minion ID as a string n representing a nonnegative integer of length k in base b, 
  where 2 <= k <= 9 and 2 <= b <= 10, write a function solution(n, b) which 
  returns the length of the ending cycle of the algorithm above starting with n.""" 
  previous_states = [] # should be [n]?
  while n not in previous_states:
    previous_states += [n]
    k = len(n)
    x = "".join(sorted(n, reverse=True)) #This might be confusing, but this is the right sort order as defined in the problem, when you think about it.
    y = "".join(sorted(n))
    z = based_subtraction(x,y,base).zfill(k)
    n = z
  return list(reversed(previous_states)).index(n)+1


print(solution('1211', 10), solution('210022', 3))

"""Input:
solution.solution('1211', 10)
Output:
    1

Input:
solution.solution('210022', 3)
Output:
    3
    """