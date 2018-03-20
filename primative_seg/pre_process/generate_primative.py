# import h5pyprovider
import numpy as np
import pickle
import os
import sys
from pointTriangleDistance import pointTriangleDistance

BASE_DIR = os.path.abspath(__file__+"/../")
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)  # model
sys.path.append(os.path.dirname(BASE_DIR))  # model
sys.path.append(ROOT_DIR)  # provider
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from show3d_balls import showpoints

''' 
    Input position x, y, z's unit is meter
    x_theta, y_theta, z_theta is arc angle
    This is the function would form tranformation matrix for any generated cannonical shapes
    including rotation along x, y, z axis and translation to another position
'''
def get_transfer_matrix( x, y, z, x_theta, y_theta, z_theta):
    x_rotation = np.asarray([
        [1 ,0, 0, 0],
        [0 , np.cos(x_theta), -np.sin(x_theta), 0],
        [0 , np.sin(x_theta), np.cos(x_theta), 0],
        [0 ,0, 0, 1]
    ], dtype=float)
    y_rotation = np.asarray([
        [np.cos(y_theta), 0, np.sin(y_theta), 0],
        [0, 1, 0, 0],
        [-np.sin(y_theta), 0, np.cos(y_theta), 0],
        [0, 0, 0, 1]
    ], dtype=float)
    z_rotation = np.asarray([
        [np.cos(z_theta), -np.sin(z_theta), 0, 0],
        [np.sin(z_theta), np.cos(z_theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    translation = np.asarray([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ], dtype=float)
    return x_rotation.dot(y_rotation).dot(z_rotation).dot(translation)

''' 
    Generate cube surface, pick position from 6 faces separately 
    Each faces we sample the point from a uniform grid, then add some random noise to the position
    however, the point would be still strictly on the cube's surface.
    keep_prob means the probability that we would not skip this point.
    x_theta, etc. is arc angle
'''

def gen_cube_surface(center_x, center_y, center_z, l, w, h, x_theta, y_theta, z_theta,
                     unit_density = 1000, keep_prob = 1):
    point_cloud = []
    den_step = 1.0 / np.power(unit_density,(1/3.0))
    # generate a transformation matrix
    tran_mat = get_transfer_matrix(center_x, center_y, center_z, x_theta, y_theta, z_theta)
    for i in np.arange(-l/2.0, l/2.0, den_step):
        for j in np.arange(-w / 2.0, w / 2.0, den_step):
            if np.random.uniform(0, 1) <= keep_prob:
                # add some random noise to x, y
                point_cloud.append([min(l,i + np.random.uniform(0,den_step)),
                                    min(w, j + np.random.uniform(0, den_step)), -h / 2, 1])
            if np.random.uniform(0, 1) <= keep_prob:
                point_cloud.append([min(l, i + np.random.uniform(0, den_step)),
                                    min(w, j + np.random.uniform(0, den_step)), h / 2, 1])
    for i in np.arange(-l/2.0, l/2.0, den_step):
        for k in np.arange(-h / 2.0, h / 2.0, den_step):
            if np.random.uniform(0, 1) <= keep_prob:
                # add some random noise to y, z
                point_cloud.append([min(l, i + np.random.uniform(0, den_step)), -w / 2,
                                    min(h, k + np.random.uniform(0, den_step)),  1])
            if np.random.uniform(0, 1) <= keep_prob:
                point_cloud.append([min(l, i + np.random.uniform(0, den_step)), w / 2,
                                    min(h, k + np.random.uniform(0, den_step)), 1])
    for j in np.arange(-w/2.0, w/2.0, den_step):
        for k in np.arange(-h / 2.0, h / 2.0, den_step):
            if np.random.uniform(0, 1) <= keep_prob:
                # add some random noise to x, z
                point_cloud.append([-l/2, min(w, j + np.random.uniform(0, den_step)),
                                    min(h, k + np.random.uniform(0, den_step)), 1])
            if np.random.uniform(0, 1) <= keep_prob:
                point_cloud.append([l/2, min(w, j + np.random.uniform(0, den_step)),
                                    min(h, k + np.random.uniform(0, den_step)), 1])
    #   apply homogeneous transfromation
    point_cloud = tran_mat.dot(np.transpose(point_cloud))
    return np.transpose(point_cloud[:3,:])

'''
    Generate sphere surface, first pick u and a angle uniformly 
    please refer to http://mathworld.wolfram.com/SpherePointPicking.html
    We also add noise to u and the angle, however the points would be still strictly on the sphere
    x_theta, etc. is arc angle
'''
def gen_sphere_surface(center_x, center_y, center_z, r, unit_density = 1000, keep_prob = 1):
    point_cloud = []
    # step of u
    den_u_step =  1.0 / np.power(unit_density,(1/3.0)) / r
    # step of angle
    den_angle_step = 2 * np.pi / np.power(unit_density,(2/3.0)) / r ** 2
    # print den_u_step, den_angle_step
    tran_mat = get_transfer_matrix(center_x, center_y, center_z, 0, 0, 0)
    for u in np.arange(-1, 1, den_u_step):
        for theta in np.arange(0, 2 * np.pi, den_angle_step):
            if np.random.uniform(0, 1) <= keep_prob:
                #   add
                ur = min(1,u + np.random.uniform(0, den_u_step))
                thetar = min(2* np.pi,theta + np.random.uniform(0, den_angle_step))
                point_cloud.append([r * np.power((1-np.power(ur,2)),0.5) * np.cos(thetar),
                                    r * np.power((1-np.power(ur,2)),0.5) * np.sin(thetar),
                                    r * ur, 1])
    point_cloud = np.asarray(point_cloud, dtype=float)
    #   apply homogeneous transfromation
    point_cloud = tran_mat.dot(np.transpose(point_cloud))
    return np.transpose(point_cloud[:3,:])

'''
    Generate triangle_pyramid surface, first pick points in the bounding cube,
    then, we save all these points if their minimal distance to one of the closest surface is within a threshold
    We also add noise to u and the angle, however the points would be still strictly on the sphere
    x_theta, etc. is arc angle
'''
def gen_triangle_pyramid(center_x, center_y, center_z, x_theta, y_theta, z_theta,
                          a, b, c, d, threshold = 0.1 ,unit_density = 1000, keep_prob = 1):
    point_cloud = []
    den_step = 1.0 / np.power(unit_density, (1 / 3.0))
    tran_mat = get_transfer_matrix(center_x, center_y, center_z, x_theta, y_theta, z_theta)
    pre_tran_mat = get_transfer_matrix(0, 0, 0, np.pi / 3.0, np.pi / 3.0, np.pi / 3.0)
    # print pre_tran_mat
    a = np.transpose(pre_tran_mat.dot(np.transpose(a))[:3])
    b = np.transpose(pre_tran_mat.dot(np.transpose(b))[:3])
    c = np.transpose(pre_tran_mat.dot(np.transpose(c))[:3])
    d = np.transpose(pre_tran_mat.dot(np.transpose(d))[:3])
    # 4 triangle surfaces
    TRIabc = np.asarray([a, b, c])
    TRIabd = np.asarray([a, b, d])
    TRIacd = np.asarray([a, c, d])
    TRIbcd = np.asarray([b, c, d])
    for i in np.arange(min(a[0], b[0], c[0], d[0]), max(a[0], b[0], c[0], d[0]), den_step):
        for j in np.arange(min(a[1], b[1], c[1], d[1]), max(a[1], b[1], c[1], d[1]), den_step):
            for k in np.arange(min(a[2], b[2], c[2], d[2]), max(a[2], b[2], c[2], d[2]), den_step):
                if np.random.uniform(0, 1) <= keep_prob:
                    # add random noise to x, y, z
                    ir = min(max(a[0], b[0], c[0], d[0]) ,i + np.random.uniform(0, den_step))
                    jr = min(max(a[1], b[1], c[1], d[1]) ,j + np.random.uniform(0, den_step))
                    kr = min(max(a[2], b[2], c[2], d[2]) ,k + np.random.uniform(0, den_step))
                    p = np.asarray([ir,jr,kr])
                    # calculate the point and surface's distance, find the minimum
                    dis_min = np.min([pointTriangleDistance(TRIabc, p)[0],
                                  pointTriangleDistance(TRIabd, p)[0],
                                  pointTriangleDistance(TRIacd, p)[0],
                                  pointTriangleDistance(TRIbcd, p)[0]])
                    if dis_min <= threshold:
                        point_cloud.append([ir, jr, kr, 1])
    point_cloud = np.asarray(point_cloud, dtype=float)
    #   apply homogeneous transfromation
    point_cloud = tran_mat.dot(np.transpose(point_cloud))
    return np.transpose(point_cloud[:3, :])

'''
    calculate l2 distance if need to be sparse
'''
def is_sparse(As, B, R):
    for A in As:
        if np.linalg.norm(np.asarray(A) - np.asarray(B)) < R: return False
    return True

'''
    function to generate individual scene
    L, W, H is the scene's space range
    ratio is the geometric scale of shape to the space range
    min_num and max_num are minimal number of shapes and maximal number of shapes
    unit_density is the sample density per unit volume
    keep_prob is the probability the sample point would be kept
    Sparse is to indicate whether we allowed two shapes to be close or even intersect
'''
def gen_scene(L, W, H, ratio, min_num, max_num, unit_density = 1000, keep_prob = 0.95, sparse=True):
    point_cloud = np.array([], dtype=np.float32).reshape(0,3)
    labels = np.array([], dtype=np.float32).reshape(0)
    centers = []
    # largest r
    R = np.min([L,W,H]) * ratio
    for i in range(np.random.randint(min_num, max_num)):
        r = np.random.uniform(R/1.5, R)
        choice = np.random.uniform(0,1)
        count = 0
        while True:
            if count > 1000: exit("cant't find a sparse solution, please reduce num of shapes or shape geometric ratio")
            count +=1
            center_x = np.random.uniform(-L / 2.0 + r, L / 2.0 - r)
            center_y = np.random.uniform(-W / 2.0 + r, W / 2.0 - r)
            center_z = np.random.uniform(-H / 2.0 + r, H / 2.0 - r)
            if not sparse: break
            # whether it's the first shape or it's distant enought to other shapes
            if len(centers) == 0 or is_sparse(centers, [center_x, center_y, center_z], R * 3):
                centers.append([center_x, center_y, center_z])
                break
        x_theta = np.random.uniform(0, 2*np.pi)
        y_theta = np.random.uniform(0, 2*np.pi)
        z_theta = np.random.uniform(0, 2*np.pi)
        # 33% to be a cube
        if choice < 1 / 3.0:
            l = 2 * r * np.random.uniform(0.3, 1)
            w = 2 * r * np.random.uniform(0.3, 1)
            h = 2 * r * np.random.uniform(0.3, 1)
            cube_point = gen_cube_surface(center_x, center_y, center_z,
                l, w, h, x_theta, y_theta, z_theta, unit_density=unit_density, keep_prob=keep_prob)
            point_cloud = np.concatenate([point_cloud, cube_point], axis= 0)
            labels = np.concatenate([labels, np.zeros(cube_point.shape[0], dtype=int)],axis=0)
        # 33% to be a sphere
        elif choice < 2 / 3.0:
            sphere_point = gen_sphere_surface(center_x, center_y, center_z,
                r * np.random.uniform(0.7, 1), unit_density=unit_density, keep_prob=keep_prob)
            point_cloud = np.concatenate([point_cloud, sphere_point], axis= 0)
            labels = np.concatenate([labels, np.ones(sphere_point.shape[0], dtype=int)], axis=0)
        else:
            # 33% to be a triangle pyramid
            a = [0, - 3**0.5 / 2.0 * r, -1 / 3.0 * r, 1]
            b = [6**0.5/3 * r, 2**0.5 / 3.0 * r, -1 / 3.0 * r, 1]
            c = [-6**0.5/3 * r, 2**0.5 / 3.0 * r, -1 / 3.0 * r, 1]
            d = [0, 0, r, 1]
            pyramid_point = gen_triangle_pyramid(center_x, center_y, center_z, x_theta, y_theta,
                 z_theta, a, b, c, d, threshold=0.02, unit_density=unit_density*1.4, keep_prob=keep_prob)
            point_cloud = np.concatenate([point_cloud, pyramid_point], axis= 0)
            labels = np.concatenate([labels, 2 * np.ones(pyramid_point.shape[0], dtype=int)], axis=0)
    return point_cloud, labels
#
'''
    generate whole dataset and save it to pickle
'''
def gen_data_set(name, num, keep_prob_min, keep_prob_max, min_num, max_num,
                 ratio, L, W, H, unit_density = 1000, sparse=True):
    data_file = name +".pickle"
    print os.path.join(ROOT_DIR, 'data/primatives/') + data_file
    cloud = []
    labels = []
    with open(os.path.join(ROOT_DIR, 'data/primatives/') + data_file, 'w') as fp:
        for i in range(num):
            keep_prob = np.random.uniform(keep_prob_min, keep_prob_max)
            cloud_point, l = gen_scene(L, W, H, ratio, min_num,
                   max_num, unit_density=unit_density, keep_prob=keep_prob, sparse=sparse)
            cloud.append(cloud_point)
            labels.append(l)
            print i
        pickle.dump(cloud,fp)
        pickle.dump(labels,fp)

if __name__=='__main__':
    # visualize one scene
    point_cloud, batch_label = gen_scene(1.5, 1.5, 3.0, 0.2, 10, 20, unit_density = 100000, keep_prob = 0.9, sparse=False)
    print "len(point_cloud)", len(point_cloud), batch_label
    c_gt = np.zeros((batch_label.shape[0], 3))
    color_list = np.asarray([[64, 224, 208], [220, 20, 60], [173, 255, 47]])
    for i in range(batch_label.shape[0]):
        c_gt[i,:] = color_list[int(batch_label[i]),:]
    showpoints(point_cloud, c_gt = c_gt, normalizecolor=False)
    # generate dataset
    # gen_data_set("prim_train_overlaps_20", 2000, 0.9, 1, 10, 20, 0.2, 1.5, 1.5, 3.0, unit_density = 100000, sparse=False)
    # gen_data_set("prim_test_overlaps_20", 500, 0.9, 1, 10, 20, 0.2, 1.5, 1.5, 3.0, unit_density = 100000, sparse=False)