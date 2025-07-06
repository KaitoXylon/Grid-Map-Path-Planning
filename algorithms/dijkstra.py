import heapq
import math

def reconstruct_path(came_from, current):
   
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current) 
    return path[::-1]

def get_neighbors(grid, node):
   
    rows, cols = grid.shape
    r, c = node
    neighbors = []
    
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in movements:
        nr, nc = r + dr, c + dc
      
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

def dijkstra(grid, start, goal):
  
    
    dist = {start: 0}

  
    came_from = {}

    
    priority_queue = [(0, start)]

   
    explored_nodes = set()

    while priority_queue:
       
        current_dist, current_node = heapq.heappop(priority_queue)

        
        if current_node in explored_nodes:
            continue

      
        explored_nodes.add(current_node)

       
        if current_node == goal:
            path = reconstruct_path(came_from, current_node)
            return {"path": path, "explored": list(explored_nodes)}

       
        for neighbor in get_neighbors(grid, current_node):
            
            new_dist = current_dist + 1

           
            if new_dist < dist.get(neighbor, math.inf):
               
                dist[neighbor] = new_dist
                
                came_from[neighbor] = current_node
                
                heapq.heappush(priority_queue, (new_dist, neighbor))

    return {"path": [], "explored": list(explored_nodes)}