"""The problem this solution solves is phrased somewhat obscurely in the problem statement. However, the problem is simple: Ignore the variable names. The idea of the problem is to make `num_buns` lists of tokens such that picking any `num_required` of said lists is guaranteed to union together to produce the full set of tokens, but picking any `num_required-1` of said lists is guaranteed NOT together to produce the full set of tokens. The tokens are arbitrary, but numbered.
  
  You want to use the smallest number of tokens, but I suspect that naturally falls out of the solution. You also need the first bunny to be holding tokens 1...n, but I also suspect that falls out of the problem statement.
  
  An obvious solution to this problem would be brute force. Another obvious solution would be some kind of dynamic programming. But this seems tantalizingly close to a closed-form-able math problem, like permutation and combination... hamming distances...binary strings representing inclusion booleans, 1..n 2^n...
  
  Eventually, by visual inspection of the (3,5) case, I realized the solution was essentially a transposition of the matrix created by combinations(n,r)."""

from itertools import combinations, product, combinations_with_replacement

def print_as_matrix(list_of_lists):
  for l in list_of_lists:
    string = ""
    for i in range(10):
      string += (str(i) if i in l else " ") + "  "
    print(string)

def verify(final_list_candidate, num_buns, num_required):
  """check for our set criteria. Important: this is maybe quite right."""
  superset = set().union(*final_list_candidate)
  #print(superset, final_list_candidate)
  #input()
  return (
    len(final_list_candidate) == num_buns and
    all([set().union(*c) == superset for c in combinations(final_list_candidate, num_required)]) and
    ( num_required == 0 or all([set().union(*c) < superset for c in combinations(final_list_candidate, num_required-1)]) )
  )

def solution(num_buns, num_required): #num_buns will always be between 1 and 9, and num_required will always be between 0 and 9 (both inclusive)
  #return list(combinations(range(num_buns), num_required))
  #Known constrains:
  if num_required == 0: return [[] for i in range(num_buns)]
  if num_required == 1: return [[0] for i in range(num_buns)]
  if num_required == num_buns: return [[x] for x in range(num_required)] #see, this really makes it seem like nCr
  #if num_required == num_buns-1: return list(map(list,combinations(range(num_buns),num_required)))
  if num_required > num_buns: return "frankly I dont know what to do here"
  #trivial, non-performant solution #actually, I got tired of trying to make this work
  max_key_count = 0
  while True:
    #print("max_key_count", max_key_count)
    for all_possible_blah in [combinations(range(max_key_count),x) for x in range(num_buns-1, max_key_count+1)]:
      #NOTE: It's very important not to debug-print the iterator all_possible_blah here (or any iterator), not even by coercing it to a list intermediately, because that would exhaust the iterator and the remaining code would not run! Also, you could convert it to a list, but then you burn a lot of memory!
      for final_list_candidate in combinations(all_possible_blah, num_buns):
        #print("flc", type(final_list_candidate), final_list_candidate) #NOTE: the FLC is just a regular list (of tuples) so it can safely be printed. However, this prints so much text as to hurt performance.
        #:
        #input(all([set().union(*c) == superset for c in combinations(final_list_candidate, num_required)]))
        #input(all([set().union(*c) < superset for c in combinations(final_list_candidate, num_required-1)]))
        if verify(final_list_candidate, num_buns, num_required): return final_list_candidate #is this the bottomneck? hmm... it could probably be streamlined easily, but that might be a waste of time.
    max_key_count+=1

def maybe_solution(num_buns, num_required):
  ll = [ [] for x in range(num_buns) ] # the list of lists
  current_list_index = 0
  #TODO: also, have to populate the first list with 1...n in order, up to some the max number of keys is bunny is going to carry.
  while current_list_index < 10:
    for i in range(current_list_index, current_list_index+num_required):
      while i >= num_buns: i-=num_buns
      ll[i].append(current_list_index)
    current_list_index+=1
  return sorted(ll)

#print_as_matrix([[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8], [0, 3, 4, 6, 7, 9], [1, 3, 5, 6, 8, 9], [2, 4, 5, 7, 8, 9]])
#exit()

#CLEAN SOLUTION BEGIN

from itertools import combinations

def filtertranspose(iterable): return [list(filter(lambda item: item is not None, l)) for l in zip(*iterable)]

def solution(num_buns, num_required):
  """The problem this solution solves is phrased somewhat obscurely in the problem statement. However, the problem is simple: Ignore the fiction behind the variable names. The idea of the problem is to make `num_buns` lists of tokens such that picking any `num_required` of said lists is guaranteed to union together to produce the full set of tokens, but picking any `num_required-1` of said lists is guaranteed NOT to union together to produce the full set of tokens. The tokens are arbitrary, but numbered. num_buns will always be between 1 and 9, and num_required will always be between 0 and 9 (both inclusive).
  
  I initially found this problem tricky, because it was so tantalizingly close to the logic and form of mathematical combination; I was sure there was a closed-form solution! (As opposed to a brute-force or dynamic-programming solution) Eventually, by visual inspection of the (3,5) case, I realized the general solution was essentially a transposition of the matrix created by combinations(n,r)."""
  #Known constrains/edge cases/degenerate cases:
  if num_required > num_buns: raise ValueError("Frankly I don't know what to do when you need more bunnies than you have bunnies.")
  if num_required == 0: return [[] for i in range(num_buns)] #This is my educated guess about what it means to require 0: the bunnies can just walk up and press the buttons, no keys (ie: empty list) required. It turns out that no test case requires this to be handled.
  #You don't need these, because our general logic handles them, but they are somewhat interesting:
  """
  if num_required == 1: return [[0] for i in range(num_buns)]
  if num_required == num_buns: return [[x] for x in range(num_required)] #see, this really makes it seem like nCr
  if num_required == num_buns-1: return list(map(list,combinations(range(num_buns),num_required))) 
  """
  return filtertranspose(
    [ [(i if ranger in c else None) for ranger in range(num_buns)] for i, c in enumerate(combinations(range(num_buns), num_buns-num_required+1))] #this populates the n choose r matrix with convenient values to be filtertransposed. Note that the num_buns-num_required+1 is just logic to reverse middle cases because of how the pattern gets laid out; not mathematically interesting.
  )

#CLEAN SOLUTION END

for i in range(10):
  for j in range(i+1):
    #print(i, j, solution(i,j))
    print(i+1, j, verify(solution(i+1,j), i+1, j), solution(i+1,j))
  input()

exit()
print(solution(2, 1), "should be:\n", [[0], [0]], "\n") #hamming distance 0. ncr = 2
print(solution(2, 2), "should be:\n", [[0], [1]], "\n") #hamming distance 1. ncr = 1
print(solution(4, 4), "should be:\n", [[0], [1], [2], [3]], "\n") #hamming distance 1. ncr = 1
print(solution(3, 2), "should be:\n", [[0, 1], [0, 2],[1, 2]], "\n") #hamming distance 1. ncr=3. This one REALLY is just the 3C2 process...
print(solution(5, 3), "should be:\n", [[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8], [0, 3, 4, 6, 7, 9], [1, 3, 5, 6, 8, 9], [2, 4, 5, 7, 8, 9]]) #hamming distance... varies. ncr=10. This is some subset of the 10C6 process... interesting... #each token appears thrice... hmm.


"""
-- Python cases -- 
Input:
solution.solution(2, 1)
Output:
    [[0], [0]]

Input:
solution.solution(4, 4)
Output:
    [[0], [1], [2], [3]]

Input:
solution.solution(5, 3)
Output:
    [[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8], [0, 3, 4, 6, 7, 9], [1, 3, 5, 6, 8, 9], [2, 4, 5, 7, 8, 9]]
"""

"""
Free the Bunny Workers
======================

You need to free the bunny workers before Commander Lambda's space station explodes! Unfortunately, the Commander was very careful with the highest-value workers -- they all work in 
separate, maximum-security work rooms. The rooms are opened by putting keys into each console, then pressing the open button on each console simultaneously. When the open button is pressed, 
each key opens its corresponding lock on the work room. So, the union of the keys in all of the consoles must be all of the keys. The scheme may require multiple copies of one key given to 
different minions.

The consoles are far enough apart that a separate minion is needed for each one. Fortunately, you have already relieved some bunnies to aid you - and even better, you were able to steal the 
keys while you were working as Commander Lambda's assistant. The problem is, you don't know which keys to use at which consoles. The consoles are programmed to know which keys each 
minion had, to prevent someone from just stealing all of the keys and using them blindly. There are signs by the consoles saying how many minions had some keys for the set of consoles. You 
suspect that Commander Lambda has a systematic way to decide which keys to give to each minion such that they could use the consoles.

You need to figure out the scheme that Commander Lambda used to distribute the keys. You know how many minions had keys, and how many consoles are by each work room.  You know that Command 
Lambda wouldn't issue more keys than necessary (beyond what the key distribution scheme requires), and that you need as many bunnies with keys as there are consoles to open the work room.

Given the number of bunnies available and the number of locks required to open a work room, write a function solution(num_buns, num_required) which returns a specification of how to 
distribute the keys such that any num_required bunnies can open the locks, but no group of (num_required - 1) bunnies can.

Each lock is numbered starting from 0. The keys are numbered the same as the lock they open (so for a duplicate key, the number will repeat, since it opens the same lock). For a given bunny, 
the keys they get is represented as a sorted list of the numbers for the keys. To cover all of the bunnies, the final solution is represented by a sorted list of each individual bunny's 
list of keys.  Find the lexicographically least such key distribution - that is, the first bunny should have keys sequentially starting from 0.

num_buns will always be between 1 and 9, and num_required will always be between 0 and 9 (both inclusive).  For example, if you had 3 bunnies and required only 1 of them to open the cell, you 
would give each bunny the same key such that any of the 3 of them would be able to open it, like so:
[
  [0],
  [0],
  [0],
]
If you had 2 bunnies and required both of them to open the cell, they would receive different keys (otherwise they wouldn't both actually be required), and your solution would be as 
follows:
[
  [0],
  [1],
]
Finally, if you had 3 bunnies and required 2 of them to open the cell, then any 2 of the 3 bunnies should have all of the keys necessary to open the cell, but no single bunny would be able to 
do it.  Thus, the solution would be:
[
  [0, 1],
  [0, 2],
  [1, 2],
]
"""