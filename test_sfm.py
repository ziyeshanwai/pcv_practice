from PIL import Image
import os
import cv2
from numpy import *
from pylab import *
import numpy as np
import pickle
import PCV.geometry.camera as camera
import PCV.geometry.homography as homography
import PCV.geometry.sfm as sfm
import PCV.localdescriptors.sift as sift
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt

def load_pickle_file(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            file = pickle.load(f)
        return file
    else:
        print("{} not exist".format(filename))

im1 = array(Image.open('./imgs/0.jpg'))
im2 = array(Image.open('./imgs/1.jpg'))

def loaddata(output_path):
    X1_list = []
    X2_list = []
    for i in range(0, 12):
        left_array = load_pickle_file(os.path.join(output_path, 'left_camera_array_{}.pkl'.format(i)))
        right_array = load_pickle_file(os.path.join(output_path, 'right_camera_array_{}.pkl'.format(i)))
        middle_ind = load_pickle_file(os.path.join(output_path, 'middle_camera_ind_{}.pkl'.format(i)))
        x1 = left_array[middle_ind, :]
        x2 = right_array[middle_ind, :]
        X1_list.extend(x1.tolist())
        X2_list.extend(x2.tolist())
    return X1_list, X2_list


def F_from_ransac(x1, x2, model, maxiter=5000, match_threshold=1e-3):
    """ Robust estimation of a fundamental matrix F from point
    correspondences using RANSAC (ransac.py from
    http://www.scipy.org/Cookbook/RANSAC).

    input: x1, x2 (3*n arrays) points in hom. coordinates. """

    import PCV.tools.ransac as ransac
    data = np.vstack((x1, x2))
    d = 20 # 20 is the original
    F, ransac_data = ransac.ransac(data.T, model,
                                   18, maxiter, match_threshold, d, return_all=True)
    return F, ransac_data['inliers']
output_path = r'\\192.168.104.99\double\2020-09-15\151104\output\debug'
X1_list, X2_list = loaddata(output_path)

X1_array = np.array(X1_list).reshape(-1, 2)
X2_array = np.array(X2_list).reshape(-1, 2)
left_array = load_pickle_file(os.path.join(output_path, 'left_camera_array_{}.pkl'.format(0)))
right_array = load_pickle_file(os.path.join(output_path, 'right_camera_array_{}.pkl'.format(0)))
middle_ind = load_pickle_file(os.path.join(output_path, 'middle_camera_ind_{}.pkl'.format(0)))
x1 = left_array[middle_ind, :]
x2 = right_array[middle_ind, :]
x1n = homography.make_homog(x1.T)
x2n = homography.make_homog(x2.T)
# x1n = homography.make_homog(left_array[middle_ind, :].T)
# x2n = homography.make_homog(right_array[middle_ind, :].T)
model = sfm.RansacModel()
while True:
    try:
        F, inliers = F_from_ransac(x1n, x2n, model, maxiter=5000, match_threshold=0.1)
    except Exception as e:
        print(e)
        continue
    # print("the length middle ind is {}".format(len(middle_ind)))
    print("inleiers is {}".format(inliers))
    fx = 640
    fy = 640
    cx = 480
    cy = 640
    K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype = np.float32)
    D = np.array([0, 0.0, -0.0, -0.00], dtype = np.float32)
    cameraMatrix = K
    # pts1_ud = x1[[41,10, 55, 86, 97, 61, 25, 17, 53], :]
    # pts2_ud = x2[[41,10, 55, 86, 97, 61, 25, 17, 53], :]
    pts1_ud = x1[inliers, :]
    pts2_ud = x2[inliers, :]
    # pts1_ud = x1[inliers, :]
    # pts2_ud = x2[inliers, :]

    F, m1 = cv2.findFundamentalMat(pts1_ud, pts2_ud)  # apparently not necessary

    # using the essential matrix can get you the rotation/translation bet. cameras, although there are two possible rotations:
    E, m2 = cv2.findEssentialMat(pts1_ud, pts2_ud, cameraMatrix, cv2.RANSAC, 0.99, 5e-3)
    Re1, Re2, t_E = cv2.decomposeEssentialMat(E)

    # recoverPose gets you an unambiguous R and t. One of the R's above does agree with the R determined here. RecoverPose can already triangulate, I check by hand below to compare results.
    K_l = cameraMatrix
    K_r = cameraMatrix
    retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E, pts1_ud, pts2_ud, cameraMatrix, distanceThresh=0.25)

    # given R,t you can  explicitly find 3d locations using projection
    M_r = np.concatenate((R, t), axis=1)
    M_l = np.concatenate((np.eye(3, 3), np.zeros((3, 1))), axis=1)
    proj_r = np.dot(cameraMatrix, M_r)
    proj_l = np.dot(cameraMatrix, M_l)
    points_4d_hom = cv2.triangulatePoints(proj_l, proj_r, np.expand_dims(X1_array, axis=1),
                                            np.expand_dims(X2_array, axis=1))
    points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
    # pts_3d = points_4d[:3, :].T
    


    # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype= np.float32)
    # P1 = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    # P2 = sfm.compute_P_from_fundamental(F)
    # P1 = np.dot(K, P1)
    # P2 = np.dot(K, P2)
    P1 = proj_l
    P2 = proj_r
    # X = sfm.triangulate(x1n[:, inliers], x2n[:, inliers], P1, P2)
    # X = sfm.triangulate(homography.make_homog(x1.T), homography.make_homog(x2.T), P1, P2)
    X = points_4d
    points_4d = X / np.tile(X[-1, :], (4, 1))
    points_3d = points_4d[:3, :].T
    pts_3d = points_3d

    cam1 = camera.Camera(P1)
    cam2 = camera.Camera(P2)
    x1p = cam1.project(X)
    x2p = cam2.project(X)
    error1 = np.mean(np.sqrt(np.sum(np.square(X1_array-x1p.T[:, :2]), axis=1)))
    error2 = np.mean(np.sqrt(np.sum(np.square(X2_array-x2p.T[:, :2]), axis=1)))
    error = (error1 + error2)/2
    print("error is {} {}".format(error1, error2))
    if error < 1.5:
        break

# print("error1 is {}".format(np.mean(np.sqrt(np.sum(np.square(x1-x1p.T[:, :2]), axis=1)))))
# print("error2 is {}".format(np.mean(np.sqrt(np.sum(np.square(x2-x2p.T[:, :2]), axis=1)))))
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], c='b', s=1, linewidths=3, marker='o', label='3d重建点')
plt.show()
figure(figsize=(16, 16))
imj = sift.appendimages(im1, im2)
# imj = vstack((imj, imj))
im_ori = imj.copy()
imshow(im_ori)

cols1 = im1.shape[1]
rows1 = im1.shape[0]
for i,ind in enumerate(inliers):
    # print("ind is {}, right_array[ind , :] is {}".format(ind, right_array[ind , :]))
    plot([x1n[0][i], x2n[0][i]+cols1],[x1n[1][i], x2n[1][i]],'c')
    # plot([x1[i][0], x2[i][0]+cols1],[x1[i][1], x2[i][1]],'c')
    # plot([x1n[ind, 0], x2n[ind, 0]+cols1],[x1n[ind, 1], x2n[ind, 1]],'c')
show()
imshow(imj)
for i in range(len(x1p[0])):
    if (0<= x1p[0][i]<cols1) and (0<= x2p[0][i]<cols1) and (0<=x1p[1][i]<rows1) and (0<=x2p[1][i]<rows1):
        plot([x1p[0][i], x2p[0][i]+cols1],[x1p[1][i], x2p[1][i]],'c')
axis('off')
show()

