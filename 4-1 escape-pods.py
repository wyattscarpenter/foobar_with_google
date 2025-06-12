from collections import deque

max_bunnies = 2000000

def bfs(graph, source, sink, parents):
    """Breadth-first search, used in the Edmonds-Karp algorithm solution below."""
    visited = [False] * len(graph)
    queue = deque([source])
    visited[source] = True
    while queue:
        u = queue.popleft()
        for v, capacity in enumerate(graph[u]):
            if not visited[v] and capacity > 0:
                visited[v] = True
                parents[v] = u
                if v == sink:
                    return True
                queue.append(v)
    return False

def solution(sources, sinks, graph):
    """Edmonds-Karp algorithm, in this case with extra preliminary logic to handle multiple sources and sinks by first supplying supersource and supersink.
    
    There exist much better and more sophisticated (therefore: more complicated) algorithms to address this-- the well-known 'maximum flow problem' in graph theory-- which this margin is sadly too small to contain."""
    #Establish placement of supersource and supersink (putting both at the end of the list, for convenience).
    #Expand matrix to handle supersource and supersink (both for their locations and references to the supersinks):
    graph.append([0 for i in graph[0]])
    supersource_index = len(graph)-1
    graph.append([0 for i in graph[0]])
    supersink_index = len(graph)-1 #we use the actual value of this later so we can't just make it a pythonic -1.
    for l in graph:
      l.extend([0,0])
    for source in sources:
        graph[supersource_index][source] = max_bunnies
    for sink in sinks:
        graph[sink][supersink_index] = max_bunnies
    
    #Edmonds-Karp proper
    parents = [-1] * len(graph)
    max_flow = 0
    while bfs(graph, supersource_index, supersink_index, parents):
        path_flow = max_bunnies
        s = supersink_index
        while s != supersource_index:
            path_flow = min(path_flow, graph[parents[s]][s])
            s = parents[s]
        max_flow += path_flow
        v = supersink_index
        while v != supersource_index:
            u = parents[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parents[v]
    return max_flow

print(solution(
    [0],
    [3],
    [ [0, 7, 0, 0],
      [0, 0, 6, 0],
      [0, 0, 0, 8],
      [9, 0, 0, 0]
    ]
))

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