#! /usr/bin/env python

# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证
# 运行指令：./01_pca_normal.py -i=/home/wjx/Desktop/3Dpointcloud_course/workspace/assignments/01-introduction/data/airplane_0001.ply
import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import argparse

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    
    #转换为numpy形式
    N = data.shape[0]
    X = data.to_numpy()
    #归一化中心
    mu = np.mean(X, axis=0)
    X_normalized = X - mu
    #计算H：cov(x,x)=dx=(x-x均值)*2数学证明
    func = np.cov if not correlation else np.corrcoef
    H = func(X_normalized, rowvar=False, bias=True)
    #np.linalg.eig计算矩阵特征值特征向量
    eigenvalues, eigenvectors = np.linalg.eig(H)
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors
def get_pca_o3d(w, v, points):
    """
        w: 特征值
        v: 特征向量
    返回：
        pca_set: o3d line set for pca visualization
    """
    # calculate centroid & variation along main axis:
    centroid = points.mean()
    projs = np.dot(points.to_numpy(), v[:,0])
    scale = projs.max() - projs.min()

    points = centroid.to_numpy() + np.vstack(
        (
            np.asarray([0.0, 0.0, 0.0]),
            scale * v.T
        )
    ).tolist()
    lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    # from the largest to the smallest: RGB
    #创建三维方阵并转化为列表[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    colors = np.identity(3).tolist()

    # build pca line set:
    pca_o3d = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    pca_o3d.colors = o3d.utility.Vector3dVector(colors)

    return pca_o3d
def get_surface_normals(pcd, points, knn=5):
    #利用open3D建立kdtree
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    # init:
    N = len(pcd.points)
    normals = []

    for i in range(N):
        # :
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], knn)
        # PCA主成分分析求normal:
        w, v = PCA(points.iloc[idx])
        normals.append(v[:, 0])

    return np.array(normals, dtype=np.float64)
def get_surface_normals_o3d(normals, points, scale=2):
    """ 为open3d创建表面几何特征集合
    Parameters
    ----------
        normals(numpy.ndarray): surface normals for each point
        points(pandas.DataFrame): points in the point cloud
        scale(float): 法向量长度

    Returns
    ----------
        surface_normals_o3d: o3d line set for surface normal visualization
    """
    # total number of points:
    N = points.shape[0]

    points = np.vstack(
        (points.to_numpy(), points.to_numpy() + scale * normals)
    )
    lines = [[i, i+N] for i in range(N)]
    colors = np.zeros((N, 3)).tolist()

    # build pca line set:
    surface_normals_o3d = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    surface_normals_o3d.colors = o3d.utility.Vector3dVector(colors)

    return surface_normals_o3d
def get_arguments():
    #从终端获取文件路径
    parser = argparse.ArgumentParser("Get PCA and surface normals for given point cloud.")
   
    #增加需要的指令
    required = parser.add_argument_group("Required")
    required.add_argument (
        "-i", dest="input", help="Input path of point cloud in ply format",
        required=True
    )
    return parser.parse_args()
def main(point_cloud_filename):
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file(point_cloud_filename)
    #open3d格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    #从电云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('[PCA Normal]: Total number of points', points.shape[0])

    #用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 2]#点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    
    #将主成分转化为o3d格式
    pca_o3d = get_pca_o3d(w, v, points)

    #计算点云每个点的法向量
    normals = get_surface_normals(point_cloud_o3d, points)
    #将法向量保存到normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    #获取表面方向量几何特征
    surface_normals_o3d = get_surface_normals_o3d(normals, points)

    #对表面法向量和PCA进行可视化
    o3d.visualization.draw_geometries([point_cloud_o3d, pca_o3d,surface_normals_o3d])
    
if __name__ == '__main__':
    arguments = get_arguments()
    main(arguments.input)

