#maybe useful:
# https://docs.python.org/2/library/fractions.html
# https://twitter.com/TivadarDanka/status/1612415086345457670
# https://math.stackexchange.com/questions/36210/is-the-limit-of-power-of-a-stochastic-matrix-still-a-stochastic-matrix


def solution(matrix):
  #So, I tried to remember a bunch of matrix tricks, in case this problem could be solved in one numpy call or something, but I didn't, so I'm just doing it the programmer way. (Which, if you think about it, is kind of more legible. But also takes up more mental space. So, whatever.)
  terminal_probs = []#not quite sure how to track this yet
  for row in matrix:
    if sum(row) == 0:
      pass #terminal case; just append to terminal probs and move on.
    else:
      transition_obs_count = sum(row)
      #... multiply terminal probs by this or something?
      #multiply denom by this or something?
  #see, one problem is that the process loops. Memoization solves? Maybe? I doubt I'm supposed to find the limit, but maybe. I think this is a stochastic matrix, so the limit would be the final state. Consider [0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0]... . The correct answer is [0, 3, 2, 9, 14]. How do we get there?
  #[0, 1/2, 
  #uh, got bored. Maybe we dynamic program from the final states? that seems like somthing we'd do. But I'm going to get coffee.
  #Wait, is this just solving a system of equations? Like, trivially? Hmm...
  # 1 = w = 3x + 2y + 1z ; x= 2w+1y ; that makes xyz the... denominators... actually, the reciprocles of the probabilities, I think. Still a problem for rings, though.
  #Rings of self to self are trivial. But for rings of one to back to its parent... actually, I'm not sure why they're even finite-precision entities. But they clearly are in the example, so I guess I need to think about it more.
  #0/14, (trivial)    3/14, 2/14, (also trivial, I guess)    9/14   (hmm)
  #[4,0,0,3,2,0] is 4/9, 3/9, 2/9 ... 3/9+(4/9(3/9)+(4/9)^2(3/9)...)
  
  #Is it, like, actually, you just need to multiply all the numbers in the matrix by large enough numbers to completely unreduce them? Like, array 2 implies that array 1 must have spit out 4+3+2=9 things into array 2 ... and then... 9 + 3+2 = 14?? or, wait, it's like, I know 9 things... actually 7 things.?. went into array 2, 4 of them came back, two went into... well, again the probability becomes a limit. The limit of staying in the cycle goes to zero, so I know if it gets in the cycle there's a 1/2 chance it goes to the other thing in array 1, 1/2 chance array 2, and therefore in array two 1/2*(3/5) and 1/2*(3/5)
    #Ah, I think that means I can compute them with intermediate results, since I just computed that intermediate result!
  #Or, wait, it's like that, but I'm just tracking the exact number of paths, so I know that... wait, no.
  import numpy
  solved = numpy.linalg.solve(matrix)
  print(solved)
  
#Observation: if a states goes back up to a previous state, that edge on that path can be removed, and the probability redistributed to the previous state, to flow outward.
#the solution requires tracking every non-cyclic path through the graph.
#We do this non-explicitly through a well-ordered recursion.
#Or, wait, should we be making a bunch more diedges in a big list and propagating them upward? Hmm... and/or removing them from the list when appropriate. Like, that would track all the possible paths, maybe, and reduce out cycles.
from fractions import Fraction
def solution(matrix, currentNodeIndex = 0, visited=[], global_probability_space=None):
  if global_probability_space==None: global_probability_space = [Fraction(1)]+ ([Fraction(0)]*(len(matrix)-1))
  if currentNodeIndex >= len(matrix): return global_probability_space # clean this up later
  node_denominator = sum(matrix[currentNodeIndex])
  for index, value in enumerate(matrix[currentNodeIndex]):
    if value==0: continue
    else:
      prob_mass = Fraction(value, node_denominator) #TODO: I think this actually has to consume probability mass, instead of just take a portion, so things with later node-internal index work right when we return up.
      global_probability_space[index]+=prob_mass*global_probability_space[currentNodeIndex] #I think this would need to be 
      if index not in visited:
        visited.append(index)
        global_probability_space = solution(matrix, index, visited, global_probability_space) ##
      #I think this won't work because it only goes left-to-right right now?
  currentNodeIndex+=1
  return global_probability_space

def lms(arg): return list(map(str, arg))

def m_print(matrix):
  print("[")
  for l in matrix:
    print("  ", lms(l))
  print("]")

#def printmap(arg): map(lambda x: print(x, sep=""), arg) #doesn't work or soemthing

def dot(vector1, vector2):
  """ so-called "dot product", perhaps better to be known as "scalar product" """
  acc = 0
  #print("DOT: ", lms(vector1), lms(vector2))
  for index, value in enumerate(vector1):
    acc+= ( vector1[index] * vector2[index] )
  return acc

def column(matrix, index):
  #print("matrix column:"); m_print(matrix); print(index)
  return [l[index] for l in matrix]

def rowdotcol(matrix1, matrix2):
  """ so-called "matrix multiplication" """
  result_matrix = [[-1]*len(matrix1) for _ in matrix2] #These might be wrong mxn. But I always do square matrices so...
  for list_index, l in enumerate(result_matrix):
    for number_index, number in enumerate(l):
      d = dot( matrix1[list_index], column(matrix2, number_index) )
      #print("result of dot", d)
      result_matrix[list_index][number_index] = d
      #print("result_matrix"); m_print(result_matrix)
      #input()
  return result_matrix

def rowdotcol(matrix1, matrix2): #machine-suggested
    """ so-called "matrix multiplication" """
    result_matrix = [[-1] * len(matrix2[0]) for _ in range(len(matrix1))]
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            d = sum(matrix1[i][k] * matrix2[k][j] for k in range(len(matrix2)))
            result_matrix[i][j] = d
    return result_matrix

from fractions import Fraction as f

#Wait... can I just use matrix multiplication? Surely the cycles prevent this, except in the limit
def solution(matrix):
  #process the matrix or whatever
  for index, row in enumerate(matrix):
    if not sum(row): matrix[index][index] = 1 #absorbing state
    else: 
      denominator = sum(row)
      for index, value in enumerate(row):
        row[index] = f(value, denominator)
  #lim i -> infinity, M^i - > absorbed state matrix
  while True:
    #print("uptop matrix:\n"); m_print(matrix)
    new_matrix = rowdotcol(matrix, matrix)
    if new_matrix == matrix:
      break
    else:
      matrix = new_matrix
  return matrix

def solution(matrix): #machine-suggested; doesn't work. May assume only one absorbing state? Or might just be completely busted.
    # Get the dimensions of the matrix
    n = len(matrix)

    # Find the absorbing states and non-absorbing states
    absorbing_states = []
    non_absorbing_states = []
    for i in range(n):
        if all(x == 0 for x in matrix[i][:i] + matrix[i][i+1:]):
            absorbing_states.append(i)
        else:
            non_absorbing_states.append(i)

    # If there are no absorbing states, return an error
    if not absorbing_states:
        return "Error: No absorbing states found."

    # Create the Q, R, and I matrices
    q = [[matrix[i][j] for j in non_absorbing_states] for i in non_absorbing_states]
    r = [[matrix[i][j] for j in absorbing_states] for i in non_absorbing_states]
    i = [[int(i == j) for j in non_absorbing_states] for i in non_absorbing_states]

    # Calculate the fundamental matrix
    f = i
    for k in range(n-1):
        f = [[sum([f[i][l]*q[l][j] for l in range(len(non_absorbing_states))]) for j in range(len(non_absorbing_states))] for i in range(len(non_absorbing_states))]

    # Calculate the absorbing probabilities
    b = [[sum([f[i][j]*r[j][k] for j in range(len(non_absorbing_states))]) for k in range(len(absorbing_states))] for i in range(len(non_absorbing_states))]

    # Get the initial state probabilities
    p_0 = [1] + [0]*(n-1)

    # Calculate the final probabilities
    p = [sum([p_0[i]*b[i][j] for i in range(len(non_absorbing_states))]) for j in range(len(absorbing_states))]

    # Return the absorbing probabilities
    return p

def inv_matrix(matrix): #machine-generated, doesn't seem to work. Out of index errors.
    """Inverts a matrix using only the standard python libraries, not numpy."""
    n = len(matrix)
    # create an identity matrix of the same size as the input matrix
    identity = [[float(i==j) for j in range(n)] for i in range(n)]
    # augment the input matrix with the identity matrix
    augmented_matrix = matrix + identity
    m_print(augmented_matrix)
    # perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        # divide the ith row by the ith element
        divisor = augmented_matrix[i][i]
        for j in range(n*2):
            augmented_matrix[j][i] /= divisor
        # subtract multiples of the ith row from the other rows to make their ith element zero
        for j in range(n):
            if i != j:
                multiplier = augmented_matrix[j][i]
                for k in range(n*2):
                    augmented_matrix[j][k] -= multiplier * augmented_matrix[i][k]
    # extract the inverted matrix from the augmented matrix
    inverted_matrix = [row[n:] for row in augmented_matrix]
    return inverted_matrix

#print(inv_matrix([[-3, 1],[5, 0]]))

def fractionize_array(array_of_floats, epsilon=0.0000000001): #naive method. O(n). May be timing out.
  #print(array_of_floats, "...")
  denom = 1 #I think we could do this with a binary search. Overshooting is somewhat easy to detect.
  while not all([((f*denom)%1)<epsilon for f in array_of_floats]): #there's some floating point approximation behind the scenes here that makes this work. Alas, it does not work well enough!
    #print(denom, [f*denom for f in array_of_floats], [((f*denom)%1)<epsilon for f in array_of_floats])
    #input()
    denom+=1
  return [int(f*denom) for f in array_of_floats] + [denom]

try:
  from fractions import gcd #python 3.4-, and also python 2, which the question evalutation area is in.
except:
  from math import gcd #python 3.5+
try:
  from math import lcm #python3.9+
except:
  from functools import reduce
  def lcm_binary(integer1, integer2):
    #from __future__ import division #would have to do this at top of file
    return int( integer1*integer2 / gcd(integer1, integer2) ) #this should be integer division in both python 2 and 3
  def lcm(*integers):
    return reduce(lcm_binary, integers)
  
from fractions import Fraction as f
def fractionize_array(array_of_floats):
  array_of_fractions = [f(x).limit_denominator() for x in array_of_floats]
  #print(array_of_fractions)
  denom = lcm(*[frac.denominator for frac in array_of_fractions])
  #print(denom)
  return [int(frac*denom) for frac in array_of_fractions] + [denom]

#Wait... can I just use matrix multiplication? Surely the cycles prevent this, except in the limit. THIS USES FLOATS FOR THE LIMIT.
def solution(matrix):
  #REMEMBER that this uses __future__ import up top of the file. #I no-longer bother to do this. Hopefully it is fine.
  #process the matrix or whatever
  absorbing_state_indices = []
  for index, row in enumerate(matrix):
    if not sum(row):
      absorbing_state_indices.append(index)
      matrix[index][index] = 1.0
    else: 
      denominator = sum(row)
      for index, value in enumerate(row):
        row[index] = float(value)/float(denominator) #could avoid this with a from __future__ import division up top, but this seems fine...
  #lim i -> infinity, M^i - > absorbed state matrix
  while True:
    #print("uptop matrix:\n"); m_print(matrix)
    new_matrix = rowdotcol(matrix, matrix)
    if new_matrix == matrix:
      break
    else:
      matrix = new_matrix
  return fractionize_array([matrix[0][i] for i in absorbing_state_indices])


print(
  solution([[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0,0], [0, 0, 0, 0, 0]]) #this produces the wrong result so... I guess one of the above things is wrong.
)

print(
  solution([[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
)

print(
  solution( [[1]*10 for i in range(9)] + [[0]*10] )
  )

"""
-- Python cases --
Input:
solution.solution([[0, 2, 1, 0, 0], [0, 0, 0, 3, 4], [0, 0, 0, 0, 0], [0, 0, 0, 0,0], [0, 0, 0, 0, 0]])
Output:
    [7, 6, 8, 21]

Input:
solution.solution([[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
Output:
    [0, 3, 2, 9, 14]
"""

"""
For example, consider the matrix m:
[
  [0,1,0,0,0,1],  # s0, the initial state, goes to s1 and s5 with equal probability
  [4,0,0,3,2,0],  # s1 can become s0, s3, or s4, but with different probabilities
  [0,0,0,0,0,0],  # s2 is terminal, and unreachable (never observed in practice)
  [0,0,0,0,0,0],  # s3 is terminal
  [0,0,0,0,0,0],  # s4 is terminal
  [0,0,0,0,0,0],  # s5 is terminal
]
So, we can consider different paths to terminal states, such as:
s0 -> s1 -> s3
s0 -> s1 -> s0 -> s1 -> s0 -> s1 -> s4
s0 -> s1 -> s0 -> s5
Tracing the probabilities of each, we find that
s2 has probability 0
s3 has probability 3/14
s4 has probability 1/7
s5 has probability 9/14
So, putting that together, and making a common denominator, gives an answer in the form of
[s2.numerator, s3.numerator, s4.numerator, s5.numerator, denominator] which is
[0, 3, 2, 9, 14].
"""

"""
Write a function solution(m) that takes an array of array of nonnegative ints representing how many times that state has gone to the next state and return an array of ints for each terminal state giving the exact probabilities of each terminal state, represented as the numerator for each state, then the denominator for all of them at the end and in simplest form. The matrix is at most 10 by 10. It is guaranteed that no matter which state the ore is in, there is a path from that state to a terminal state. That is, the processing will always eventually end in a stable state. The ore starts in state 0. The denominator will fit within a signed 32-bit integer during the calculation, as long as the fraction is simplified regularly. 
""" #10 lines of 10 cells? Don't make me laugh! #also, since python has transparent bignum, won't worry about that.

