import numpy as np

def projected_convex_hull(points, param):
    normal = param[:3]
    d = param[3]
    
    proj = points - np.dot(points, normal.reshape(3, 1)) * normal + d * normal
    
    convex_hull = []
    anchor_index = -1
    last_index = -1
        
    # Find anchor point, x coordinate is minimum
    anchor = proj[np.argmin(proj[:, 0])]
    convex_hull.append(anchor)
    anchor_index = np.argmin(proj[:, 0])
    last_index = anchor_index
        
    # Find convex hull proj by Javier's algorithm
    while True:
        next_index = -1
        for i in range(len(proj)):
            if i == last_index:
                continue
            
            if next_index == -1:
                next_index = i
                continue
            
            # Check if the point is to the left of the line formed by last_index and i
            if np.cross(proj[i] - proj[last_index], proj[next_index] - proj[last_index])[2] < 0:
                next_index = i
        
        if next_index == anchor_index:
            break
        
        convex_hull.append(proj[next_index])
        last_index = next_index        
       
    #print(len(convex_hull), "points in convex hull")    

    return np.array(convex_hull)