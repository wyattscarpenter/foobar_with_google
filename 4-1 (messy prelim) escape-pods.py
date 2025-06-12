"""My commentary: This is a well-known problem in graph theory, known as a flow network, described in https://en.wikipedia.org/wiki/Flow_network. However, I didn't read that page, and instead implemented the following algorithm using dynamic programming: start at all the exits. Set the capacity as the capacity of all edges into the exits, but only if it's lower than the current max capacity. Remove current nodes from the set (this, if done right, handles cycles). Repeat until at entrances.

Do you have to consider possible branching points in the bunny allocation? Initially I thought no, since you can account for nodes along the path as a class, and thus consider all in an out flow to them at the same time. And, trying to brute-force this would be high-time-complexity (maybe that's fine because it's 50 rooms max?) However, there are weird cases like:

 * Initial node (infinite bunnies)
/10 \1
\1  /10
  * Exit node (only receives one bunny!)

The problem statement doesn't say the graph is acyclic so I assume it can have cycles, which is obviously something to take care of. I initially supposed that removing nodes visiting in the algorithm from further consideration is enough, because it's not like bunnies flowing back into the node are going to be helpful. They could maybe be helpful, but not in """

"""I initially designed a dynamic programming solution to this problem, but then realized my algorithm sucked and I should just use Edmonds-Karp. There are better algorithms than Edmonds-Karp, even, but Edmonds-Karp is relatively simple.

Edmonds-Karp is O(VE^2), which is possibly bad because the only constrain of the problem is the number of rooms, 50, meaning there are maybe approximately 50^2 edges... so that could end up 50*(50^2)^2. So maybe I should use Dinitz's algorithm, which is O(V^2E), for 50^2*50^2... that's still pretty bad, though. Well, let's see if it matters!"""

def __edmonds_karp(graph, s, t):
    # Initialize flow to zero
    flow = 0

    while True:
        # Run a breadth-first search (bfs) to find the shortest s-t path.
        # Use 'pred' to store the edge taken to get to each vertex,
        # so we can recover the path afterwards.
        q = deque()
        q.append(s)
        pred = [None] * len(graph)
        while q:
            cur = q.popleft()
            for e in graph[cur]:
                if pred[e[1]] is None and e[1] != s and e[0] > e[2]:
                    pred[e[1]] = e
                    q.append(e[1])

        if pred[t] is not None:
            # We found an augmenting path. See how much flow we can send.
            df = float('inf')
            e = pred[t]
            while e is not None:
                df = min(df, e[0] - e[2])
                e = pred[e[1]]

            # Update edges by that amount.
            e = pred[t]
            while e is not None:
                e[2] += df
                e[3][e[4]][2] -= df
                e = pred[e[1]]

            flow += df
        else:
            # No augmenting path was found, so we're done.
            break

    return flow

from collections import deque, namedtuple

Edge = namedtuple('Edge', ['cap', 'flow', 's', 't', 'rev'])

def __edmonds_karp_named_tuple(graph, s, t):
    # Initialize flow to zero
    flow = 0

    while True:
        # Run a breadth-first search (bfs) to find the shortest s-t path.
        # Use 'pred' to store the edge taken to get to each vertex,
        # so we can recover the path afterwards.
        q = deque()
        q.append(s)
        pred = [None] * len(graph)
        while q:
            cur = q.popleft()
            for e in graph[cur]:
                if pred[e.t] is None and e.t != s and e.cap > e.flow:
                    pred[e.t] = e
                    q.append(e.t)

        if pred[t] is not None:
            # We found an augmenting path. See how much flow we can send.
            df = float('inf')
            e = pred[t]
            while e is not None:
                df = min(df, e.cap - e.flow)
                e = pred[e.s]

            # Update edges by that amount.
            e = pred[t]
            while e is not None:
                e = e._replace(flow=e.flow + df)
                e.rev = e.rev._replace(flow=e.rev.flow - df)
                e = pred[e.s]

            flow += df
        else:
            # No augmenting path was found, so we're done.
            break

    return flow

from collections import deque

class FlowEdge:
    def __init__(self, u, v, capacity):
        self.u = u
        self.v = v
        self.capacity = capacity
        self.flow = 0
        self.residual = None
    
    def get_residual(self):
        if self.residual is None:
            self.residual = FlowEdge(self.v, self.u, 0)
            self.residual.residual = self
        return self.residual
    
    def remaining_capacity(self):
        return self.capacity - self.flow
    
    def augment(self, bottleneck):
        self.flow += bottleneck
        self.residual.flow -= bottleneck
    
    def __str__(self):
        return f"({self.u} -> {self.v} | {self.flow}/{self.capacity})"
    
    def __repr__(self):
        return str(self)


def __edmonds_karp_class(graph, s, t):
    # Initialize flow to zero
    flow = 0

    while True:
        # Run a breadth-first search (bfs) to find the shortest s-t path.
        # Use 'pred' to store the edge taken to get to each vertex,
        # so we can recover the path afterwards.
        q = deque()
        q.append(s)
        pred = [None] * len(graph)
        while q:
            cur = q.popleft()
            for e in graph[cur]:
                if pred[e.v] is None and e.remaining_capacity() > 0:
                    pred[e.v] = e
                    q.append(e.v)

        if pred[t] is not None:
            # We found an augmenting path. See how much flow we can send.
            df = float('inf')
            e = pred[t]
            while e is not None:
                df = min(df, e.remaining_capacity())
                e = pred[e.u]

            # Update edges by that amount.
            e = pred[t]
            while e is not None:
                e.augment(df)
                e = pred[e.u]

            flow += df
        else:
            # No augmenting path was found, so we're done.
            break

    return flow


from collections import deque

def edmonds_karp(graph, s, t):
    flow = 0

    while True:
        # Run a breadth-first search (bfs) to find the shortest s-t path.
        # Use 'pred' to store the edge taken to get to each vertex,
        # so we can recover the path afterwards.
        q = deque()
        q.append(s)
        pred = [None] * len(graph)
        while q:
            cur = q.popleft()
            for e in graph[cur]:
                if pred[e[1]] is None and e[1] != s and e[0] > e[2]:
                    pred[e[1]] = e
                    q.append(e[1])

        if pred[t] is not None:
            # We found an augmenting path. See how much flow we can send.
            df = float('inf')
            e = pred[t]
            while e is not None:
                df = min(df, e[0] - e[2])
                e = pred[e[1]]

            # Update edges by that amount.
            e = pred[t]
            while e is not None:
                e[2] += df
                e[3][e[4]][2] -= df
                e = pred[e[1]]

            flow += df
        else:
            # No augmenting path was found, so we're done.
            break

    return flow


"""
solution
  #Let's do it iteratively. Might be easier recursively but whatever. Can't be naively recursive due to sharing capacity up the chain.
  current_capacity=max_bunnies
  current_nodes = set(exits)
  while list(current_nodes) != entrances:
    input(list(current_nodes))
"""  

from collections import deque, namedtuple

FlowEdge = namedtuple('FlowEdge', ['v', 'capacity', 'flow', 'residual'])

def __solution1(adj_matrix, sources, sinks):
    # Initialize flow to zero
    flow = 0

    # Convert adjacency matrix to list of edges
    edges = []
    for u, neighbors in enumerate(adj_matrix):
        for v, capacity in enumerate(neighbors):
            if capacity > 0:
                edges.append(FlowEdge(v, capacity, 0, None))

    while True:
        # Run a breadth-first search (bfs) to find the shortest s-t path.
        # Use 'pred' to store the edge taken to get to each vertex,
        # so we can recover the path afterwards.
        q = deque()
        pred = [None] * len(adj_matrix)
        for s in sources:
            q.append(s)
            pred[s] = FlowEdge(s, float('inf'), 0, None)

        while q:
            cur = q.popleft()
            for e in edges:
                if e.v != cur or e.remaining_capacity() <= 0:
                    continue
                if pred[e.v] is None:
                    pred[e.v] = e
                    q.append(e.v)

        # We found an augmenting path. See how much flow we can send.
        bottleneck = float('inf')
        for t in sinks:
            e = pred[t]
            while e.v != e.residual.v:
                bottleneck = min(bottleneck, e.remaining_capacity())
                e = pred[e.residual.v]
            bottleneck = min(bottleneck, e.remaining_capacity())

        if bottleneck == float('inf'):
            # No augmenting path was found, so we're done.
            break

        # Update edges by that amount.
        for t in sinks:
            e = pred[t]
            while e.v != e.residual.v:
                e.augment(bottleneck)
                e = pred[e.residual.v]
            e.augment(bottleneck)

        flow += bottleneck

    return flow

from collections import deque, namedtuple

def __edmonds_karp(sources, sinks, graph):
    Edge = namedtuple('Edge', ['capacity', 'flow', 'source', 'sink', 'reverse'])
    n = len(graph)
    edges = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if graph[i][j] != 0:
                rev_edge = Edge(0, 0, j, i, None)
                edge = Edge(graph[i][j], 0, i, j, rev_edge)
                edges[i].append(edge)
                edges[j].append(rev_edge._replace(reverse=edge))

    max_flow = 0
    while True:
        q = deque([sources[0]])
        prev = [None] * n
        while q:
            cur = q.popleft()
            for e in edges[cur]:
                if prev[e.sink] is None and e.capacity > e.flow:
                    prev[e.sink] = e
                    q.append(e.sink)
        if prev[sinks[0]] is not None:
            df = float('inf')
            e = prev[sinks[0]]
            while e is not None:
                df = min(df, e.capacity - e.flow)
                e = prev[e.source]
            e = prev[sinks[0]]
            while e is not None:
                e = e._replace(flow=e.flow + df)
                e.reverse = e.reverse._replace(flow=e.reverse.flow - df)
                e = prev[e.source]
            max_flow += df
        else:
            break
    return max_flow

from collections import deque, namedtuple

def __edmonds_karp(sources, sinks, graph):
    Edge = namedtuple('Edge', ['capacity', 'flow', 'source', 'sink', 'reverse'])
    n = len(graph)
    edges = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if graph[i][j] != 0:
                rev_edge = Edge(0, 0, j, i, None)
                edge = Edge(graph[i][j], 0, i, j, rev_edge)
                edges[i].append(edge)
                edges[j].append(rev_edge._replace(reverse=edge))

    max_flow = 0
    while True:
        q = deque([sources[0]])
        prev = [None] * n
        while q:
            cur = q.popleft()
            for e in edges[cur]:
                if prev[e.sink] is None and e.capacity > e.flow:
                    prev[e.sink] = e
                    q.append(e.sink)
        if prev[sinks[0]] is not None:
            df = float('inf')
            e = prev[sinks[0]]
            while e is not None:
                df = min(df, e.capacity - e.flow)
                e = prev[e.source]
            e = prev[sinks[0]]
            while e is not None:
                e = e._replace(flow=e.flow + df)
                e.reverse = e.reverse._replace(flow=e.reverse.flow - df)
                e = prev[e.source]
            max_flow += df
        else:
            break
    return max_flow

from collections import deque, namedtuple

Edge = namedtuple('Edge', ['capacity', 'flow', 'source', 'sink', 'reverse'])

def edmonds_karp(graph, source, sink):
    flow = 0
    while True:
        queue = deque(); queue.append(source)
        predecessor = [None] * len(graph)
        while queue:
            current = queue.popleft()
            for edge in graph[current]:
                if predecessor[edge.sink] is None and edge.sink != source and edge.capacity > edge.flow:
                    predecessor[edge.sink] = edge; queue.append(edge.sink)
        if predecessor[sink] is not None:
            delta_flow = float('inf')
            edge = predecessor[sink]
            while edge is not None:
                delta_flow = min(delta_flow, edge.capacity - edge.flow)
                edge = predecessor[edge.source]
            edge = predecessor[sink]
            while edge is not None:
                edge = edge._replace(flow=edge.flow + delta_flow)
                edge.reverse = edge.reverse._replace(flow=edge.reverse.flow - delta_flow)
                edge = predecessor[edge.source]
            flow += delta_flow
        else:
            break
    return flow


max_bunnies = 2000000

#Use max-bunnies to create a supersource and supersink nodes?

solution = edmonds_karp

print(
  solution(
    [0],
    [3],
    [ [0, 7, 0, 0],
      [0, 0, 6, 0],
      [0, 0, 0, 8],
      [9, 0, 0, 0]
    ]
  )
)#6

print(
  solution(
  [0, 1],
  [4, 5],
  [ [0, 0, 4, 6, 0, 0],
    [0, 0, 5, 2, 0, 0],
    [0, 0, 0, 0, 4, 4],
    [0, 0, 0, 0, 6, 6],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
  ]
  )
)#16

"""
Escape Pods
===========

You've blown up the LAMBCHOP doomsday device and relieved the bunnies of their work duries -- and now you need to escape from the space station as quickly and as orderly as possible! The 
bunnies have all gathered in various locations throughout the station, and need to make their way towards the seemingly endless amount of escape pods positioned in other parts of the station. 
You need to get the numerous bunnies through the various rooms to the escape pods. Unfortunately, the corridors between the rooms can only fit so many bunnies at a time. What's more, many 
of the corridors were resized to accommodate the LAMBCHOP, so they vary in how many bunnies can move through them at a time. 

Given the starting room numbers of the groups of bunnies, the room numbers of the escape pods, and how many bunnies can fit through at a time in each direction of every corridor in between, 
figure out how many bunnies can safely make it to the escape pods at a time at peak.

Write a function solution(entrances, exits, path) that takes an array of integers denoting where the groups of gathered bunnies are, an array of integers denoting where the escape pods are 
located, and an array of an array of integers of the corridors, returning the total number of bunnies that can get through at each time step as an int. The entrances and exits are disjoint 
and thus will never overlap. The path element path[A][B] = C describes that the corridor going from A to B can fit C bunnies at each time step.  There are at most 50 rooms connected by the 
corridors and at most 2000000 bunnies that will fit at a time.

For example, if you have:
entrances = [0, 1]
exits = [4, 5]
path = [
  [0, 0, 4, 6, 0, 0],  # Room 0: Bunnies
  [0, 0, 5, 2, 0, 0],  # Room 1: Bunnies
  [0, 0, 0, 0, 4, 4],  # Room 2: Intermediate room
  [0, 0, 0, 0, 6, 6],  # Room 3: Intermediate room
  [0, 0, 0, 0, 0, 0],  # Room 4: Escape pods
  [0, 0, 0, 0, 0, 0],  # Room 5: Escape pods
]

Then in each time step, the following might happen:
0 sends 4/4 bunnies to 2 and 6/6 bunnies to 3
1 sends 4/5 bunnies to 2 and 2/2 bunnies to 3
2 sends 4/4 bunnies to 4 and 4/4 bunnies to 5
3 sends 4/6 bunnies to 4 and 4/6 bunnies to 5

So, in total, 16 bunnies could make it to the escape pods at 4 and 5 at each time step.  (Note that in this example, room 3 could have sent any variation of 8 bunnies to 4 and 5, such as 2/6 
and 6/6, but the final solution remains the same.)

Input:
solution.solution([0], [3], [[0, 7, 0, 0], [0, 0, 6, 0], [0, 0, 0, 8], [9, 0, 0, 0]])
Output:
    6

Input:
solution.solution([0, 1], [4, 5], [[0, 0, 4, 6, 0, 0], [0, 0, 5, 2, 0, 0], [0, 0, 0, 0, 4, 4], [0, 0, 0, 0, 6, 6], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
Output:
    16
"""