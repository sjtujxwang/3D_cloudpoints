#！ usr/bin/env python
#kdtree的具体实现，包括构建和查找
#构建自己的KDtree，并实现k-r查找
import time
import random
import math
import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet

# Node类，Node是tree的基本组成元素
class Node:
    #进行初始化
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis
        self.value = value
        self.left = left
        self.right = right
        self.point_indices = point_indices
    #判断是否是叶节点
    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output

# 功能：构建树之前需要对value进行排序，同时对一个的key的顺序也要跟着改变
# 输入：
#     key：键
#     value:值
# 输出：
#     key_sorted：排序后的键
#     value_sorted：排序后的值
def sort_key_by_vale(key, value):
    assert key.shape == value.shape
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted


def axis_round_robin(axis, dim):
    if axis == dim-1:
        return 0
    else:
        return axis + 1

# 功能：通过递归的方式构建树
# 输入：
#     root: 树的根节点
#     db: 点云数据
#     point_indices：排序后的键
#     axis: scalar
#     leaf_size: scalar
# 输出：
#     root: 即构建完成的树
def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)
        
    # 作业1
    # 屏蔽开始

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  #按照值对点索引进行排序 可以将排序算法改成寻找中值算法，也可以寻找平均值以代替排序算法建立非严格排序算法
        middle_left_idx=math.ceil(point_indices_sorted.shape[0]/2)-1 #根节点左孩是第几个 a.shape[0] 矩阵a第一维度的长度
        middle_left_point_idx=point_indices_sorted[middle_left_idx] #在排好序的数组中间位置的id
        middle_left_point_value=db[middle_left_point_idx, axis]#root的值

        middle_right_idx=middle_left_idx+1 #右孩id
        middle_right_point_idx=point_indices_sorted[middle_right_idx]
        middle_right_point_value=db[middle_right_point_idx, axis]

        root.value=(middle_left_point_value+middle_right_point_value)/2
        
        #get the split position
        root.left=kdtree_recursive_build(root.left,db,point_indices_sorted[0:middle_right_idx],axis_round_robin(axis,dim=db.shape[1]),leaf_size)
        root.right=kdtree_recursive_build(root.right,db,point_indices_sorted[middle_right_idx:],axis_round_robin(axis,dim=db.shape[1]),leaf_size)
        # 屏蔽结束
    return root


# 功能：翻转一个kd树
# 输入：
#     root：kd树
#     depth: 当前深度
#     max_depth：最大深度
def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1

# 功能：构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
# 输入：
#     db_np：原始数据
#     leaf_size：scale
# 输出：
#     root：构建完成的kd树
def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


# 功能：通过kd树实现knn搜索，即找出最近的k个近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set：搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False

#compare query to every point inside the leaf, put into the result set
    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1) #求范数,求查询点到叶节点的距离
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])#将对应点距离和id计入结果
        return False

    # 作业2
    # 提示：仍通过递归的方式实现搜索
    # 屏蔽开始

    if query[root.axis]<=root.value:
        kdtree_knn_search(root.left,db,result_set,query)#q[axis] inside the partition
        if math.fabs(query[root.axis]-root.value)<result_set.worstDist():
            kdtree_knn_search(root.right,db,result_set,query)#|q[axis]-splitting_value|<w 
    else:
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis]-root.value)<result_set.worstDist():
            kdtree_knn_search(root.left,db,result_set,query)

    # 屏蔽结束

    return False

# 功能：通过kd树实现radius搜索，即找出距离radius以内的近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set:搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False
    
    # 作业3
    # 提示：通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis]<=root.value:
        kdtree_radius_search(root.left,db,result_set,query)
        if math.fabs(query[root.axis]-root.value)<result_set.worstDist():
            kdtree_radius_search(root.right,db,result_set,query)
    else:
        kdtree_radius_search(root.right,db,result_set,query)
        if math.fabs(query[root.axis]-root.value)<result_set.worstDist():
            kdtree_radius_search(root.left,db,result_set,query)
    # 屏蔽结束

    return False



def main():
    # configuration
    leaf_size = 4
    db_size = 64
    dim = 3
    k = 8
    r = 0.372
    D=3
    k_brute_time=0
    r_brute_time=0
    kdtree_knn_time=0
    kdtree_rnn_time=0


    pointcloud_in = np.fromfile(str("/home/wjx/Desktop/homework/000000.bin"), dtype=np.float32, count=-1).reshape([-1, 4])
    db_np = pointcloud_in[:,[0,1,2]]

    root = kdtree_construction(db_np, leaf_size=leaf_size)
    
    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    print("...........")
    print("tree max depth: %d" % max_depth[0])

    #查询8个临近点   自己实现的比/numy暴力搜索的
    # k暴力搜索
    for _ in range(100):
        query=np.random.rand(D) 
        #01--knn brute-force as baseline
        begin_t=time.time()
        dists = np.linalg.norm(db_np - query, axis=1)
        sorting_idx = np.argsort(dists)
        brute_force_result = {i for i in sorting_idx[:k]}
        k_brute_time += time.time() - begin_t
    print("k_brute_time:")
    print(k_brute_time)
    print("k_brute_result:")
    print(brute_force_result)
    # kdtree knn搜索
    for _ in range(100):
        query=np.random.rand(D) 
        begin_t=time.time()
        knn_result_set = KNNResultSet(capacity=k)
        kdtree_knn_search(root, db_np, knn_result_set, query)
        knn_result = {i.index for i in knn_result_set.dist_index_list}
        kdtree_knn_time += time.time() - begin_t
    print("kdtree_knn_time:")
    print(kdtree_knn_time)
   
    #r暴力搜索
    for _ in range(100):
        begin_t=time.time()
        dists = np.linalg.norm(db_np - query, axis=1)
        brute_force_result = {i for i, d in enumerate(dists) if d <= r}
        r_brute_time += time.time()-begin_t
    print("r_brute_time:")
    print(r_brute_time)
    # print("brute_force_result:")
    # print(brute_force_result)  

    #kdtree rnn搜索
    for _ in range(100):
        begin_t=time.time()
        query=np.random.rand(D)
        rnn_result_set = RadiusNNResultSet(radius = r)
        kdtree_radius_search(root, db_np, rnn_result_set, query)
        rnn_result = {i.index for i in rnn_result_set.dist_index_list}
        kdtree_rnn_time += time.time() - begin_t 
    print("kdtree_rnn_time:")
    print(kdtree_rnn_time)
    
    print("knn_grade:")
    print(kdtree_knn_time/k_brute_time)
    print("rnn_grade:")
    print(kdtree_rnn_time/r_brute_time)
    
    



    #02--
    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # knn_search(root, db_np, result_set, query)
    #
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])
    #
    #
    # print("Radius search:")
    # query = np.asarray([0, 0, 0])
    # result_set = RadiusNNResultSet(radius = 0.5)
    # radius_search(root, db_np, result_set, query)
    # print(result_set)


if __name__ == '__main__':
    main()