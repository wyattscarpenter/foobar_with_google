def x_(tup,i):
  return (tup[0]+i,tup[1])
def y_(tup,j):
  return (tup[0],tup[1]+j)
def xy_(tup,i,j):
  return (tup[0]+i,tup[1]+j)
def xys_(tup,arr_ijs):
  possibilities = []
  for arr in arr_ijs:
    i = arr[0]
    j = arr[1]
    possibilities += [(tup[0]+i, tup[1]+j), (tup[0]-i, tup[1]+j), (tup[0]+i, tup[1]-j), (tup[0]-i, tup[1]-j),
      (tup[0]+j, tup[1]+i), (tup[0]-j, tup[1]+i), (tup[0]+j, tup[1]-i), (tup[0]-j, tup[1]-i)
    ]
  checked_possibilities = [ p for p in possibilities if (p[0] >= 0 and p[0] < 8) and (p[1] >= 0 and p[1] < 8) ]
  return checked_possibilities

def checked_knights_moves(from_point):
  #could probably do some elegant functional map-reduce thing here but I'm not really feeling it.
  points = from_point

square_side_len = 8 #I never use this, but whatever

def solution(src, dst):
  #I'm not keeping track of which is which, here, and I don't care to start.
  src_tuple = (src//8, src%8)
  dst_tuple = (dst//8, dst%8)
  moves = 0
  old_states = [src_tuple]
  while dst_tuple not in old_states:
    moves+=1
    new_states = []
    for p in old_states:
     new_states += xys_(p,[[2,1]])
    old_states = new_states
  return moves

print(solution(0, 1)) #should be 3
print(solution(19, 36)) #should be 1