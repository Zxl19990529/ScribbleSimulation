import numpy as np
from Bezier import Bezier
from graphutils import build_graph,build_kruskal,DFS,drawgraph_inorder
import os
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import cv2
color_map = [[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
                              [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                              [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
                              [0,192,0], [128,192,0], [0,64,128]]


# def get_useless_bondaries(mask):


def get_bezier_points(points):
    # points: (n,2) n is the points number, 2 is h and w.
    # return: scribble  (t_values,2)   t_values points.
    y,x = points[:,0],points[:,1] # (n,)
    y_min = y.min()
    y_max = y.max()
    h = y_max-y_min
    
    x_min = x.min()
    x_max = x.max()
    w = x_max-x_min
    
    t_values = np.arange(0,1,1/(1*(w+h)))
    scribble = Bezier.Curve(t_values,points)
    return scribble

def eur_distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
# def cal_path_length(curr_path):
    
def find_the_longest_path(all_paths):
    longest_path = []
    max_depth = 0
    for path in all_paths:
        curr_depth = 0
        for i in range(len(path)-1):
            p1 = int(path[i][0]),int(path[i][1])
            p2 = int(path[i+1][0]),int(path[i+1][1])
            curr_depth += eur_distance(p1,p2)
        if curr_depth > max_depth:
            max_depth = curr_depth
            longest_path = path
    return longest_path

def find_random_path(all_paths):
    length = []
    choices = np.arange(0,len(all_paths),1)
    for path in all_paths:
        curr_depth = 0
        for i in range(len(path)-1):
            p1 = int(path[i][0]),int(path[i][1])
            p2 = int(path[i+1][0]),int(path[i+1][1])
            curr_depth += eur_distance(p1,p2)
        length.append(curr_depth)
    length = np.array(length)
    # probability = []
    probability = length/length.sum()
    # print(probability)
    selected_id = int(np.random.choice(choices,size=1,p=probability))
    return all_paths[selected_id]

def get_dfs_points(skeleton_mask):
    '''skeleton_mask: binary skeleton mask, usually processed by Medial axis transformation, shape of (h,w)
    return: max_depth, shape of (n,2), the longest path of the skeleton.
    '''
    vertices,edges,neighbours = build_graph(skeleton_mask) # convert binary skeleton mask to graph
    if len(vertices) == 1:
        return [vertices]
    vertices, new_edges, new_neighbours = build_kruskal(vertices,edges,neighbours) # generate the minimum kruskal tree to remove the circle,# vertices are (x,y)
    # vertices: list,[(x1,y1),(x2,y2)...]
    # edges: set, {(vertex1,vertex2,distance),....} the vertex1 is the index in vertices.
    # neighbours: dict, { (x1,y1):[(xi,yi),(xj,yj)....] }
    vertices_array = np.array(vertices) # n,2
    x,y = vertices_array[:,0],vertices_array[:,1]
    
    y_min = y.min()
    y_max = y.max()
    h = y_max-y_min
    
    x_min = x.min()
    x_max = x.max()
    w = x_max-x_min
    if h<w:
        vertices = sorted(vertices,key=lambda x:x[0])
        start_vec = vertices[0]
    else:
        vertices = sorted(vertices,key=lambda x:x[1])
        start_vec = vertices[0]
    
    start_vec = tuple(reversed(start_vec))
    all_paths = DFS(start_vec,new_neighbours,visited=set(),curr_path=[],all_paths=[])
    max_depth_path = find_the_longest_path(all_paths)    
    return max_depth_path

def get_random_points(skeleton_mask):
    '''skeleton_mask: binary skeleton mask, usually processed by Medial axis transformation, shape of (h,w)
    return: a random path, shape of (n,2), a path of the skeleton, whose possiblity is decided by its length.
    '''
    vertices,edges,neighbours = build_graph(skeleton_mask) # convert binary skeleton mask to graph
    if len(vertices) == 1:
        return [vertices]
    vertices, new_edges, new_neighbours = build_kruskal(vertices,edges,neighbours) # generate the minimum kruskal tree to remove the circle,# vertices are (x,y)
    # vertices: list,[(x1,y1),(x2,y2)...]
    # edges: set, {(vertex1,vertex2,distance),....} the vertex1 is the index in vertices.
    # neighbours: dict, { (x1,y1):[(xi,yi),(xj,yj)....] }
    vertices_array = np.array(vertices) # n,2
    x,y = vertices_array[:,0],vertices_array[:,1]
    
    y_min = y.min()
    y_max = y.max()
    h = y_max-y_min
    
    x_min = x.min()
    x_max = x.max()
    w = x_max-x_min
    if h<w:
        vertices = sorted(vertices,key=lambda x:x[0])
        start_vec = vertices[0]
    else:
        vertices = sorted(vertices,key=lambda x:x[1])
        start_vec = vertices[0]
    
    start_vec = tuple(reversed(start_vec))
    all_paths = DFS(start_vec,new_neighbours,visited=set(),curr_path=[],all_paths=[])
    max_depth_path = find_random_path(all_paths)
    return max_depth_path

def decide_path_or_boundary(binary_mask):
    '''
    binary_mask: h,w
    return a Flag, where 0 uses the path, 1 uses the boundary.
    '''
    h,w = binary_mask.shape
    choices = [0,1]
    probability = binary_mask.sum()/(h*w)
    if probability <0.1: # too small, unecessary to draw boundaries.
        return False
    num_labels, regions, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8) # regions H*W, value denotes the i-th region.
    if num_labels > 2:
        return False
    probability = [1-probability,probability]
    flag = np.random.choice(choices,size=1,p=probability)[0]
    return bool(flag)

def decide_path_or_boundary_medical(binary_mask):
    '''
    binary_mask: h,w
    return a Flag, where 0 uses the path, 1 uses the boundary.
    '''
    h,w = binary_mask.shape
    choices = [0,1]

    # 最小外接圆
    contours,_ = cv2.findContours(binary_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #cv2.RETR_EXTERNAL 定义只检测外围轮廓
    (x, y), radius = cv2.minEnclosingCircle(contours[0])
    center = (int(x), int(y))
    radius = int(radius)

    externel_circle_mask = np.zeros((h,w,3),np.uint8)
    externel_circle_mask = cv2.circle(externel_circle_mask, center, radius, (1, 1, 1), thickness=-1)
    object_contour_fillmask = np.zeros((h,w,3),np.uint8)
    object_contour_fillmask = cv2.drawContours(object_contour_fillmask,contours,0,color=(1,1,1),thickness=-1)

    probability = object_contour_fillmask.sum()/externel_circle_mask.sum()
    if probability <0.333333: # too thin, unecessary to draw boundaries.
        return False
    if probability > 0.9: # perfect circle.
        return True
    num_labels, regions, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8) # regions H*W, value denotes the i-th region.
    if num_labels > 2:
        return False
    probability = [1-probability,probability]
    flag = np.random.choice(choices,size=1,p=probability)[0]
    return bool(flag)

def xtract_minEnclosingCircle(binary_mask):
    ''' given a binary mask return the centriod (x,y) and radius
    '''
    binary_mask[binary_mask>0] = 1
    contours,_ = cv2.findContours(binary_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #cv2.RETR_EXTERNAL 定义只检测外围轮廓
    (x, y), radius = cv2.minEnclosingCircle(contours[0])
    center = (int(x), int(y))
    radius = int(radius)
    return center,radius

def swap_xy(array):
    ''' the input array is a (n,2) shape, swap it second demsion
    ''' 
    tmp = array[:,0].copy()
    array[:,0] = array[:,1]
    array[:,1] = tmp
    return array

def compute_boundary_centroid(vertices_array):
    ''' vertices_array: (n,2) h,w not swaped .
    '''
    # cv_vertivces = swap_xy(vertices_array)
    M = cv2.moments(vertices_array)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return np.array([cx,cy])
    

def random_linear_aff(vertices_array,jit_pixel=20,center_point=None):
    ''' vertices_array:(n,2)
        center_point: use the centriod point or the center point
    '''
    cy,cx = center_point
    y = vertices_array[:,0]
    x = vertices_array[:,1]
    slop_array = (y-cy) / (x-cx+1e-3)
    bias_array = y - slop_array*x

    delta = np.random.randint(low=-jit_pixel+1,high=jit_pixel-1,size=len(x))

    alpha = np.arctan(slop_array)
    delta_x = delta * np.cos(alpha)
    delta_y = delta * np.sin(alpha)
    
    new_x = x + delta_x
    new_y = y + delta_y

    new_vertices = np.zeros_like(vertices_array)
    new_vertices[:,0] = new_y
    new_vertices[:,1] = new_x
    new_vertices.astype(np.uint8)
    
    return new_vertices
def find_largest_connected_area(binary_mask):
    ''' binary_mask # h,w
    '''
    num_labels, regions, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8) # regions H*W, value denotes the i-th region.
    largest_area = 0
    largest_regionid = None
    for region_id  in np.unique(regions):
        if region_id == 0:
            continue
        region = (regions == region_id)
        if np.sum(region) >=largest_area:
            largest_area = np.sum(region)
            largest_regionid = region_id
    # if largest_area <=2:
    clean_binary_mask = np.zeros_like(binary_mask)
    clean_binary_mask[regions == largest_regionid] = 1
    return clean_binary_mask


def get_random_boundary(binary_mask,jitter_pixel=20,vis_path=None):
    h,w = binary_mask.shape
    if vis_path:
        plt.figure(figsize=(19.2,10.8))
        plt.subplot(2,2,1),plt.imshow(binary_mask),plt.title('Mask')
    # 开运算 先腐蚀再膨胀，去掉边缘毛刺，保留整形态。
    kernel = np.ones((jitter_pixel,jitter_pixel), np.uint8) # 开运算的kernel，这里核的尺寸也限制了随机仿射的程度，防止超出mask。
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
    if vis_path:
        plt.subplot(2,2,2),plt.imshow(binary_mask),plt.title('Eroded Mask') 
    binary_mask = find_largest_connected_area(binary_mask) # after eroded, some small areas should be removed.
    boundary = mark_boundaries(binary_mask,binary_mask,color=(1,1,1),mode='inner') # this may return with a float64 map, containing 0.0039
    boundary = boundary.astype(np.uint8)
    if vis_path:
        plt.subplot(2,2,3),plt.imshow(boundary*255),plt.title('Boundary points')
    # plt.savefig('vis.png')

    vertices,edges,neighbours = build_graph(boundary[:,:,0],thin=False) # convert binary skeleton mask to graph
    # vertices.pop(-1)
    vertices_array = np.array(vertices) # n,2
    # center_array = np.mean(vertices_array,axis=0,dtype=np.int32) # (2,) compute the center point
    centriod_array = compute_boundary_centroid(vertices_array)# (2,) compute the centriod point
    jittered_vertices_array = random_linear_aff(vertices_array,jit_pixel=20,center_point = centriod_array)
    #---debug visualization---#
    canva_vertice = np.zeros_like(boundary)
    for i in range(len(vertices_array)-1):
        curr_point = vertices[i]
        next_point = vertices[i+1]
        y1,x1 = curr_point
        y2,x2 = next_point
        cv2.circle(canva_vertice,(x1,y1),3,(255,255,255),-1)
        cv2.putText(canva_vertice, str(i), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1)

        cv2.circle(canva_vertice,(x2,y2),3,(255,255,255),-1)
        cv2.putText(canva_vertice, str(i+1), (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1)
        cv2.line(canva_vertice, (x1,y1), (x2,y2), (255,255,255),1)

        jitter_curr_point = jittered_vertices_array[i]
        next_jitter_curr_point = jittered_vertices_array[i+1]
        jittered_y1,jittered_x1 = jitter_curr_point
        jittered_y2,jittered_x2 = next_jitter_curr_point
        cv2.circle(canva_vertice,(jittered_x1,jittered_y1),3,(0,255,255),-1)
        cv2.circle(canva_vertice,(jittered_x2,jittered_y2),3,(0,255,255),-1)
        cv2.line(canva_vertice, (jittered_x1,jittered_y1), (jittered_x2,jittered_y2), (0,255,255),1)

    # center_y,center_x = center_array
    # cv2.circle(canva_vertice,(center_x,center_y),4,(255,0,255),-1)
    center_y,center_x = centriod_array 
    cv2.circle(canva_vertice,(center_x,center_y),4,(255,255,0),-1)
    if vis_path:
        plt.subplot(2,2,4),plt.imshow(canva_vertice),plt.title('Affined boundary points')
        plt.tight_layout()
        if vis_path is not None:
            plt.savefig(vis_path)
        plt.close()

    jittered_vertices_array = swap_xy(jittered_vertices_array) # to adapt the opencv format h-w -> w-h

    
    return jittered_vertices_array


def get_random_boundary_v2(binary_mask,jitter_pixel=20,vis_path=None):
    h,w = binary_mask.shape
    plt.figure(figsize=(19.2,10.8))
    plt.subplot(2,2,1),plt.imshow(binary_mask),plt.title('Mask')
    # 开运算 先腐蚀再膨胀，去掉边缘毛刺，保留整形态。
    kernel = np.ones((jitter_pixel,jitter_pixel), np.uint8) # 开运算的kernel，这里核的尺寸也限制了随机仿射的程度，防止超出mask。
    binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
    plt.subplot(2,2,2),plt.imshow(binary_mask),plt.title('Eroded Mask') 
    binary_mask = find_largest_connected_area(binary_mask) # after eroded, some small areas should be removed.
    boundary = mark_boundaries(binary_mask,binary_mask,color=(1,1,1),mode='inner') # this may return with a float64 map, containing 0.0039
    boundary = boundary.astype(np.uint8)
    plt.subplot(2,2,3),plt.imshow(boundary*255),plt.title('Boundary points')
    # plt.savefig('vis.png')

    vertices,edges,neighbours = build_graph(boundary[:,:,0],thin=False) # convert binary skeleton mask to graph
    # vertices, new_edges, new_neighbours = build_kruskal(vertices,edges,neighbours) # generate the minimum kruskal tree to remove the circle,# vertices are (x,y)
    start_vec = tuple(reversed(vertices[0]))
    all_paths = DFS(start_vec,neighbours,visited=set(),curr_path=[],all_paths=[])
    max_depth_path = find_the_longest_path(all_paths)
    # vertices.pop(-1)
    vertices_array = np.array(max_depth_path) # n,2
    vertices_array = swap_xy(vertices_array)
    # center_array = np.mean(vertices_array,axis=0,dtype=np.int32) # (2,) compute the center point
    centriod_array = compute_boundary_centroid(vertices_array)# (2,) compute the centriod point
    jittered_vertices_array = random_linear_aff(vertices_array,jit_pixel=20,center_point = centriod_array)
    #---debug visualization---#
    canva_vertice = np.zeros_like(boundary)
    for i in range(len(vertices_array)-1):
        curr_point = vertices_array[i]
        next_point = vertices_array[i+1]
        y1,x1 = curr_point
        y2,x2 = next_point
        cv2.circle(canva_vertice,(x1,y1),3,(255,255,255),-1)
        cv2.putText(canva_vertice, str(i), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1)

        cv2.circle(canva_vertice,(x2,y2),3,(255,255,255),-1)
        cv2.putText(canva_vertice, str(i+1), (x2,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 1)
        cv2.line(canva_vertice, (x1,y1), (x2,y2), (255,255,255),1)

        jitter_curr_point = jittered_vertices_array[i]
        next_jitter_curr_point = jittered_vertices_array[i+1]
        jittered_y1,jittered_x1 = jitter_curr_point
        jittered_y2,jittered_x2 = next_jitter_curr_point
        cv2.circle(canva_vertice,(jittered_x1,jittered_y1),3,(0,255,255),-1)
        cv2.circle(canva_vertice,(jittered_x2,jittered_y2),3,(0,255,255),-1)
        cv2.line(canva_vertice, (jittered_x1,jittered_y1), (jittered_x2,jittered_y2), (0,255,255),1)

    # center_y,center_x = center_array
    # cv2.circle(canva_vertice,(center_x,center_y),4,(255,0,255),-1)
    center_y,center_x = centriod_array
    cv2.circle(canva_vertice,(center_x,center_y),4,(255,255,0),-1)
    plt.subplot(2,2,4),plt.imshow(canva_vertice),plt.title('Affined boundary points')
    plt.tight_layout()
    if vis_path is not None:
        plt.savefig(vis_path)
    plt.close()

    jittered_vertices_array = swap_xy(jittered_vertices_array) # to adapt the opencv format h-w -> w-h

    
    return jittered_vertices_array

def xtract_region_by_color(img,color=[0,0,0],return_binary=True):
    r''' given a RGB img, return a binary mask reflecting the specific color.
    '''
    # mask = np.zeros_like(img)[:,:,0]
    mask = (img[:, :, 0]==color[0])&(img[:, :, 1]==color[1])&(img[:, :, 2]==color[2])
    if return_binary:
        return mask
    else:
        mask = mask[:,:,None].repeat(3,-1)
        return mask

def xtract_region_by_circle(img,centroid=(255,255),r=250,return_binary=True):
    r''' given a RGB img, return a binary mask reflecting the specific color.
    '''
    # mask = np.zeros_like(img)[:,:,0]
    mask = cv2.circle(np.zeros_like(img),centroid,r,color=(1,1,1),thickness=-1)
    # mask = ~mask
    if return_binary:
        return mask[:,:,0]
    else:
        return mask