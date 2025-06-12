# There are many ways to solve this problem. The way I decided to do it relies on the fact that in the limit, taking an absorbing Markov chain matrix to the power of n converges to its final state. That is, if you matrix-multiple m with itself enough (theoretically, an infinite number of times), it becomes the result matrix we are interested in. In this solution, we go to the limit of float precision, and this turns out to be enough. To produce the integer answers, we round the floats to the nearest fraction a smidge, and this seems to work fine. Because the problem constraint includes the verbiage "the denominator will fit within a signed 32 bit integer during the calculation as long as the fraction is simplified regularly", it may be possible to prove that the precision bounds we have set on this problem always produce an exact, correct result, but I have not attempted such a proof. Due to the process we use, the runtime of this algorithm is (if not outclassed by some other bottleneck like actually performing the matrix multiplies) proportional to the precision of your floating point number type--see https://docs.python.org/2/tutorial/floatingpoint.html#representation-error and https://en.wikipedia.org/wiki/Double-precision_floating-point_format for more information on modern floats, if need be.
# As I say, this is a well-known problem in discrete math. See, for example, https://en.wikipedia.org/wiki/Absorbing_Markov_chain. The "right way to do it" for this problem is to read the final probabilities out of a matrix B, where B = NR, N = (I-Q)^(--1), I is an identity matrix, and R and Q are subsets of the original matrix. So, that's fine, but the ^(-1) means I would have to write a matrix inverter, which means I would need to write Gauss-Jordan elimination, and also matrix multiplication, and at that point I already had matrix multiplication, so I decided to go with the matrix multiplication numerical approximation method in this file. I don't even know if this "right way" would be fast. Probably. These matrices are probably not computationally intensive to invert. The naive Gauss I would write is probably O(n^3), but n is small. Also, come to think of it, my matrix multiplication algorithm is also naive and O(n^3).
# There are also probably graph theory (possibly dynamic programming, or Bellman-Ford-like) ways to solve this problem. I thought fairly hard about them, but I couldn't think of a good way to resolve arbitrary cycles of cycles when I really got down to it: it seemed simple, but when I tried visiting the graph in the way I planned, and doing the accounting that I thought should work, I never got the right probabilities by the end.

#from __future__ import division # I could have used this, but decided not to.

def dot(vector1, vector2):
  """ so-called "dot product", perhaps better to be known as "scalar product" """
  return sum( [vector1[index] * vector2[index] for index, value in enumerate(vector1)] )

def column(matrix, index):
  """ Returns a list that is the column of the matrix of the index referred to by index. Like a row, but since the rows are lists natively, it's one little step less trivial. """
  return [l[index] for l in matrix]

def rowdotcol(matrix1, matrix2):
  """ so-called "matrix multiplication" """
  result_matrix = [[-1]*len(matrix1) for _ in matrix2] #matrix1 and matrix2 might be in the wrong order, conceptually, here-- but I always do square matrices in the current application. In fact, I always do the same matrix multiplied by itself, so this avoids loosing track of which is which!
  for list_index, l in enumerate(result_matrix):
    for number_index, number in enumerate(l):
      d = dot( matrix1[list_index], column(matrix2, number_index) )
      result_matrix[list_index][number_index] = d
  return result_matrix

try:
  from fractions import gcd #python 3.4-, and also python 2, which the question evaluation area is in.
except:
  from math import gcd #python 3.5+
try:
  from math import lcm #python3.9+
except:
  from functools import reduce
  def lcm_binary(integer1, integer2):
    return int( integer1*integer2 / gcd(integer1, integer2) ) # This properly produces integer division in both python 2 and 3.
  def lcm(*integers):
    return reduce(lcm_binary, integers)
  
from fractions import Fraction as f
def fractionize_array(array_of_floats):
  array_of_fractions = [f(x).limit_denominator() for x in array_of_floats]
  denom = lcm(*[frac.denominator for frac in array_of_fractions])
  return [int(frac*denom) for frac in array_of_fractions] + [denom]

# Just uses matrix multiplication. Cycles prevent this, except in the limit. THIS USES FLOATS FOR THE LIMIT.
def solution(matrix):
  # Process the matrix into standard mathematical form, using floats, noting its absorbing_state_indices so we can return those later:
  absorbing_state_indices = []
  for index, row in enumerate(matrix):
    if not sum(row):
      absorbing_state_indices.append(index)
      matrix[index][index] = 1.0
    else: 
      denominator = sum(row)
      for index, value in enumerate(row):
        row[index] = float(value)/float(denominator)
  #lim i -> infinity, M^i - > absorbed state matrix
  while True:
    new_matrix = rowdotcol(matrix, matrix)
    if new_matrix == matrix:
      break
    else:
      matrix = new_matrix
  # The matrix now complete, we just read out the probabilities and represent them as integers.
  return fractionize_array([matrix[0][i] for i in absorbing_state_indices])