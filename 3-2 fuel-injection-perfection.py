def new(n):
  return [n+1, n-1] + ([n//2] if n%2==0 else [])

def ss(s,e): set_add_and_return_success(s, e)

def set_add_and_return_success(s, e):
  success = e not in s
  s.add(e)
  return success

def solution(pellet_count_string): # a number up to 309 digits long
  n = int(pellet_count_string) #since this is python, it just automatically uses a bignum internally.
  #but 309 is way too many digits to bottom-up problem solve from 1 by making a huge dag or whatever. Or maybe it's fine if you do it smartly? Eh doesn't seem promising.
  #i imagine, if the number is really big, divide by two is more useful than +/- 1. Why? Because you just go way further.
  #powers of two are surely important in this problem, but I don't know if they dominate. Factors of two as well. Perhaps it's all about factoring? Modular arithmetic? Mod 2?
  #I decided to just do it the flood fill method. A flood fill in the sky!
  #I assume this is better than the flood fill from 0 method, but it's not really clear why it would be.
  s = set([n])
  move_count = 0
  current_queue = [n]
  next_queue = []
  while 1 not in current_queue:
    #print("current queue:", current_queue)
    move_count+=1
    #I didn't understand where my code was going wrong, so I thought, what if I were to just make this one list comprehension? But then I ran into one of the bad corner cases of list comprehensions, so the following line of code is even wronger:
    #current_queue = [newbie for newbie in [new(m) for m in current_queue if ss(s,m)] for ]
    for m in current_queue:
      for newbie in new(m):
        #print("newbie %d element of set %s?", newbie, s)
        if newbie not in s:
          s.add(newbie)
          next_queue.append(newbie)
    current_queue = next_queue
    next_queue = []
    if current_queue == []: exit("current_queue emptied itself, somehow...")
  return move_count

print(solution('1'))
print(solution('4'))
print(solution('15'),solution('4'))
"""
-- Python cases -- 
Input:
solution.solution('15')
Output:
    5

Input:
solution.solution('4')
Output:
    2
    """