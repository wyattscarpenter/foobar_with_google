from __future__ import division
from itertools import combinations, product, combinations_with_replacement, chain, permutations
from math import factorial #note: the more-general math.perm was introduced only in python 3.8, so we can't use it for python 2.7

def print_starfields(list_of_starfields):
  for starfield in list_of_starfields:
    for l in starfield:
      print(l,",", sep="")
    print()

def fl(n):
  """Shorthand for factorial."""
  return factorial(n)

def comb(n, r):
  """Like math.comb, but here in python 2."""
  return fl(n) // ( fl(r) * fl(n - r) ) 

def comb_with_replacement(n, r):
  """Like math.comb, but with replacement."""
  return comb(n+r-1,r)

def solution(w, h, s):
  "The obvious way to solve this is with some sort of brute force / memoized traversal. But is there a math way... our old friend combinations? Since this is about rotating things, (actually, swapping things) perhaps group theory? Perhaps... sorting the array for our memoized traversal? Symmetry groups? Closed form solution, somehow, from removing the number of permutations/symmetry groups (the final state, factorial)? Well... combination is already permutation where order doesn't matter... so there must be some kind of 2d combination one can do... combinations with replacement of a group ignoring the permutations of its rows and columns... permutations all told (product?) minus permutation of those permutations..."
  """Star grid standardization means that the width and height of the grid will always be between 1 and 12, inclusive. And while there are a variety of celestial bodies in each grid, the number of states of those bodies is between 2 and 20, inclusive. The solution can be over 20 digits long, so return it as a decimal string.  The intermediate values can also be large, so you will likely need to use at least 64-bit integers."""
  
  #loc = list(product(product(range(s), repeat=w), repeat=h))
  print("combinations(range(s), w)", list(combinations_with_replacement(range(s), w)))
  loc = list(combinations_with_replacement(combinations_with_replacement(range(s), w), h))
  print_starfields(loc)
  number_of_products = (w**h)**s
  number_of_final_states_unfiltered = s**(w*h)
  print(number_of_products)
  print(len(loc))
  print(number_of_final_states_unfiltered)
  print(len(set(loc)))
  input()
  number_of_permutations = factorial(0) #uh... what to put in here...

  #Let's go step by step...
  number_of_final_states_unfiltered =w*h*s #this is  # s**(w*h)#factorial() #maybe factorial this, or something, for the reasons below? #IMPORTANT: Maybe use something like factorial (of all possible w*h-length s-lists?) to generate these instead of exponents so filtering is more straightforward?
  filtered_for_w = number_of_final_states_unfiltered / factorial(w*h) #this approach fails I think because we are double-excluding permutations of all zeros, I think. But something like this is bound to work.... either by cleverly modulating this feature, or by... radically double-counting the unfiltered states?
  print("filtered_for_w", filtered_for_w)
  filtered_for_h = filtered_for_w / factorial(h)
  print("filtered_for_h", filtered_for_h)

#Uh, I think it's the product, but you have to filter the RHSs to not produce repeats. Hmm... maybe that's too hard.
#perhaps the solutuion is O(s) and you have to do a little figuring for each possible symbol? hmm...
  
  return str(len(loc))

def transposed_permutations(x): return [tuple(zip(*p)) for p in permutations(zip(*x))]

def solution(w, h, s): # MUST return string of digits for the number.
  print("naive state estimation:", s**(w*h))
  #OK... let's do the computationally bad option to figure it out...
  #One way to do it:
  all_possible_cart_prods = set(product(product(range(s), repeat=w), repeat=h))
  #print_starfields(all_possible_cart_prods)
  #input((w, h, s))
  true_set = set()
  while(all_possible_cart_prods): #this removes from the set during the loop
    e = all_possible_cart_prods.pop()
    old_set = set() #This business is to try to fully populate a set of all possible permutations of e, which we do by adding permuations until the set stops growing
    new_set = set([e])
    the_new_stuff = set([e])
    while new_set - old_set: #If this is true, then there is at least some new!
      old_set |= new_set
      new_set = set()
      for x in the_new_stuff:
        new_set |= set(permutations(x)) | set(transposed_permutations(x)) #Now that this includes the transposed permuts, it works!
      the_new_stuff = new_set - old_set
      #print("old_set", old_set, "new_set", new_set)
    #OK, now that the loop is done, it now has all possible permutations of e
    if not old_set & true_set: #set intersection /\, ie this finds if the process has created anything that wasn't already in the old set
      true_set.add(e)
    all_possible_cart_prods -= old_set #the minus-assign here... really helps the runtime of the algorithm lmao
    the_new_stuff = new_set - old_set
  return str(len(true_set))
  #The above solution only passes formal tests 1,2 & 4, as they are trivial. However, I made it to see if I can spot patterns in the low numbers.
    
    #bfs/all possible permutations, populate that set.
  return str(
    comb_with_replacement(comb_with_replacement(s, h), w)
  )

print("Test 2:", solution(2, 2, 2), "should be", "7")
input("Go on?")
print("Test 1:", solution(2, 3, 4), "should be", "430")
'''
print("Test 3:", solution(1, 11, 3), "should be", "78") #CWR works
print("Test 4:", solution(1, 1, 20), "should be", "20") #CWR works, as does mere comb, as does prod, and... straight multiplication.
print("Test 7:", solution(1, 10, 2), "should be", "11") #CWR works
'''
input("Now test all in range(1,5)?")

for param_tuple in product(range(1,5), repeat=3):
  print(param_tuple, solution(*param_tuple))
#Inefficient test output:
"""
(1, 1, 1) 1
(1, 1, 2) 2
(1, 1, 3) 3
(1, 1, 4) 4
(1, 2, 1) 1
(1, 2, 2) 3
(1, 2, 3) 6
(1, 2, 4) 10
(1, 3, 1) 1
(1, 3, 2) 4
(1, 3, 3) 10
(1, 3, 4) 20
(1, 4, 1) 1
(1, 4, 2) 5
(1, 4, 3) 15
(1, 4, 4) 35
(2, 1, 1) 1
(2, 1, 2) 3
(2, 1, 3) 6
(2, 1, 4) 10
(2, 2, 1) 1
(2, 2, 2) 7
(2, 2, 3) 27
(2, 2, 4) 76
(2, 3, 1) 1
(2, 3, 2) 13
(2, 3, 3) 92
(2, 3, 4) 430
(2, 4, 1) 1
(2, 4, 2) 22
(2, 4, 3) 267
(2, 4, 4) 1996
(3, 1, 1) 1
(3, 1, 2) 4
(3, 1, 3) 10
(3, 1, 4) 20
(3, 2, 1) 1
(3, 2, 2) 13
(3, 2, 3) 92
(3, 2, 4) 430
(3, 3, 1) 1
(3, 3, 2) 36
(3, 3, 3) 738
(3, 3, 4) 8240
(3, 4, 1) 1
(3, 4, 2) 87
(3, 4, 3) 5053
"""
#Ahh... it's just our old friend https://oeis.org/A353585 !

# # More code I turned out to not need: 
# from collections import OrderedDict
#def has_no_repetitions_in_tuple(x):
#  return tuple(OrderedDict.fromkeys(x)) == x
# def distinct_partitions_of_integer(i):
  # """The name of this mathematical concept may be misleading. It's actually the partitions of i where the partitions are not allowed to include any repeated elements. Eg, 3+1+1 is not allowed, but 3+2 is allowed. See https://oeis.org/A000009 for more details"""
  # return [p for p in partitions_of_integer(i) if has_no_repetitions_in_tuple(p)]
# def flatten_2d_list(list_of_lists):
  # return [x for l in list_of_lists for x in l]
# dpois = distinct_partitions_of_integer(i)
  # dpois_set = set(dpois)
  # dpois_flat = flatten_2d_list(dpois)


"""
Disorderly Escape
=================

Oh no! You've managed to free the bunny workers and escape Commander Lambdas exploding space station, but Lambda's team of elite starfighters has flanked your ship. If you dont jump 
to hyperspace, and fast, youll be shot out of the sky!

Problem is, to avoid detection by galactic law enforcement, Commander Lambda planted the space station in the middle of a quasar quantum flux field. In order to make the jump to hyperspace, 
you need to know the configuration of celestial bodies in the quadrant you plan to jump through. In order to do *that*, you need to figure out how many configurations each quadrant could 
possibly have, so that you can pick the optimal quadrant through which youll make your jump. 

There's something important to note about quasar quantum flux fields' configurations: when drawn on a star grid, configurations are considered equivalent by grouping rather than by 
order. That is, for a given set of configurations, if you exchange the position of any two columns or any two rows some number of times, youll find that all of those configurations are 
equivalent in that way -- in grouping, rather than order.

Write a function solution(w, h, s) that takes 3 integers and returns the number of unique, non-equivalent configurations that can be found on a star grid w blocks wide and h blocks tall where 
each celestial body has s possible states. Equivalency is defined as above: any two star grids with each celestial body in the same state where the actual order of the rows and columns do not 
matter (and can thus be freely swapped around). Star grid standardization means that the width and height of the grid will always be between 1 and 12, inclusive. And while there are a variety 
of celestial bodies in each grid, the number of states of those bodies is between 2 and 20, inclusive. The solution can be over 20 digits long, so return it as a decimal string.  The 
intermediate values can also be large, so you will likely need to use at least 64-bit integers.

For example, consider w=2, h=2, s=2. We have a 2x2 grid where each celestial body is either in state 0 (for instance, silent) or state 1 (for instance, noisy).  We can examine which grids are 
equivalent by swapping rows and columns.

00
00

In the above configuration, all celestial bodies are "silent" - that is, they have a state of 0 - so any swap of row or column would keep it in the same state.

00 00 01 10
01 10 00 00

1 celestial body is emitting noise - that is, has a state of 1 - so swapping rows and columns can put it in any of the 4 positions.  All four of the above configurations are equivalent.

00 11
11 00

2 celestial bodies are emitting noise side-by-side.  Swapping columns leaves them unchanged, and swapping rows simply moves them between the top and bottom.  In both, the *groupings* are the 
same: one row with two bodies in state 0, one row with two bodies in state 1, and two columns with one of each state.

01 10
01 10

2 noisy celestial bodies adjacent vertically. This is symmetric to the side-by-side case, but it is different because there's no way to transpose the grid.

01 10
10 01

2 noisy celestial bodies diagonally.  Both have 2 rows and 2 columns that have one of each state, so they are equivalent to each other.

01 10 11 11
11 11 01 10

3 noisy celestial bodies, similar to the case where only one of four is noisy.

11
11

4 noisy celestial bodies.

There are 7 distinct, non-equivalent grids in total, so solution(2, 2, 2) would return 7.

Languages
=========

To provide a Java solution, edit Solution.java
To provide a Python solution, edit solution.py

Test cases
==========
Your code should pass the following test cases.
Note that it may also be run against hidden test cases not shown here.

-- Java cases -- 
Input:
Solution.solution(2, 3, 4)
Output:
    430

Input:
Solution.solution(2, 2, 2)
Output:
    7

-- Python cases --
Input:
solution.solution(2, 3, 4)
Output:
    430

Input:
solution.solution(2, 2, 2)
Output:
    7

Use verify [file] to test your solution and see how it does. When you are finished editing your code, use submit [file] to submit your answer. If your solution passes the test cases, it will be removed from your home folder.
"""
