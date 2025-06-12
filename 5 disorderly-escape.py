from __future__ import division
from math import factorial
from fractions import Fraction
try:
  from fractions import gcd #python 3.4-, and also python 2, which the question evaluation area is in.
except:
  from math import gcd #python 3.5+

try:
  from math import prod #not available in python 2
except:
  from functools import reduce
  def prod(i): return reduce( lambda x, y: x*y, i)

def partitions_of_integer(i, helper_argument=1): #"P"
  yield (i,)
  for j in range(helper_argument, i//2 + 1):
    for p in partitions_of_integer(i-j, j):
      yield (j,) + p

def sum_of_gcds(p, q): #"S"
  return sum([gcd(i, j) for i in p for j in q])

def N(t):
  """There is no concise name for this function so I've elected to just call it N. What is does is compute a combinatoric measure involving the distinct partitions of the input integer--basically, a weighted function of counting elements of the p-tuple.
  
  N(p) = Product_{distinct parts x in p} x^m(x)*m(x)!, where m(x) = multiplicity of x in p."""
  t_as_set = set(t) # We use the set to get a listing of each distinct part once
  return prod( [ x**t.count(x)*factorial(t.count(x)) for x in t_as_set] )

def solution(w, h, s):
  """Returns a string of decimal digits as the answer to the problem.
  
  After some trial-and-error, I realized that this problem is just our old friend, https://oeis.org/A353585, _number of inequivalent matrices over Z/nZ, modulo permutations of rows and columns, of size r X c_. Therefore, this problem admits of a purely-numerical (though not technically "closed-form") solution-- which is good, because if it had to create all the states it had to report, it would take a long time!
  
  Parameter constraints from the problem: Star grid standardization means that the width and height of the grid will always be between 1 and 12, inclusive. And while there are a variety of celestial bodies in each grid, the number of states of those bodies is between 2 and 20, inclusive. The solution can be over 20 digits long, so return it as a decimal string.  The intermediate values can also be large, so you will likely need to use at least 64-bit integers."""
  return str( int( sum(
    [ Fraction( s**sum_of_gcds(p, q) , (N(p)*N(q)) ) for p in partitions_of_integer(w) for q in partitions_of_integer(h) ]
  ) ) )

### Clean solution above this line. ###

print("Test 2:", solution(2, 2, 2), "should be", "7")
print("Test 1:", solution(2, 3, 4), "should be", "430")
print("Test 1':", solution(3, 2, 4), "should be", "430")
print("Test 3:", solution(1, 11, 3), "should be", "78") #CWR works
print("Test 4:", solution(1, 1, 20), "should be", "20") #CWR works, as does mere comb, as does prod, and... straight multiplication.
print("Test 7:", solution(1, 10, 2), "should be", "11") #CWR works
print("Test blah:", solution(12, 12, 20), "should be", "IDK, might time out though.") #CWR works
#hidden test-cases still fail... I'll have to test other points on the triangle, and also cross-reference my results with the possibly-working results online.

for n in range(1, 11):
  print()
  for i in range(1,4+1):
    for j in range(1,i+1):
      #print((i, j, n), solution(i, j, n))
      print(solution(i, j, n),"\t", sep="", end="")
exit()
"""
for n in range(1, 11):
  print()
  for i in range(1,4+1):
    for j in range(1,i+1):
      #print((i, j, n), solution(i, j, n))
      print(solution(j, i, n),"\t", end="")
"""
