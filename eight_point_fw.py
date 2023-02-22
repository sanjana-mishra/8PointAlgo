import math
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

def find_matching_keypoints(image1, image2):
    #Input: two images (numpy arrays)
    #Output: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    #img1: image on which we draw the epilines for the points in img2
    #lines: corresponding epilines
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def FindFundamentalMatrix(pts1, pts2):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    #todo: Normalize the points
    #finding centroid of points
    centroid1 = np.mean(pts1, axis=0)
    centroid2 = np.mean(pts2, axis=0)

    ## finding mean dists
    dist1 = []
    dist2 = []
    for i in range(pts1.shape[0]):
        dist = math.sqrt((((centroid1[0]-pts1[i, 0])**2) + ((centroid1[1]-pts1[i, 1])**2)))
        dist1.append(dist)
    mean_dist1 = np.mean(dist1, axis=0)
    #print(mean_dist1)

    for i in range(pts1.shape[0]):
        dist = math.sqrt((((centroid2[0]-pts2[i, 0])**2) + ((centroid2[1]-pts2[i, 1])**2)))
        dist2.append(dist)
    mean_dist2 = np.mean(dist2, axis=0)
    #print(mean_dist2)

    # #translating pts
    # pts1 = pts1 - centroid1
    # pts2 = pts2 - centroid2

    pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1), dtype=np.int32)))
    pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1), dtype=np.int32)))


    ##new translated pts and new centroid
    # new_centroid1 = np.mean(pts1, axis=0)
    # print(new_centroid1)
    # dist1_new = []
    # for i in range(0, 155):
    #     dist = round(math.sqrt((((new_centroid1[0]-pts1[i, 0])**2) + ((new_centroid1[1]-pts1[i, 1])**2))))
    #     dist1_new.append(dist)
    # mean_dist1_new = np.mean(dist1_new, axis=0)
    # print(mean_dist1_new)
    #
    # new_centroid2 = np.mean(pts2, axis=0)
    # print(new_centroid2)
    # dist2_new = []
    # for i in range(0, 155):
    #     dist = round(math.sqrt((((new_centroid2[0]-pts2[i, 0])**2) + ((new_centroid2[1]-pts2[i, 1])**2))))
    #     dist2_new.append(dist)
    # mean_dist2_new = np.mean(dist2_new, axis=0)
    # print(mean_dist2_new)

    scale1 = np.sqrt(2)/mean_dist1
    scale2 = np.sqrt(2)/mean_dist2

    t1 = np.array([[scale1, 0, 0],
                   [0, scale1, 0],
                   [scale1*(-1 * centroid1[0]), scale1*(-1 * centroid1[1]), 1]])
    t1 = t1.T

    t2 = np.array([[scale2, 0, 0],
                   [0, scale2, 0],
                   [scale2*(-1 * centroid2[0]), scale2*(-1 * centroid2[1]), 1]])
    t2 = t2.T

    pts1_new = (t1 @ pts1.T)
    pts2_new = (t2 @ pts2.T)

    n = pts1_new.shape[1]

    A = np.zeros((n, 9))
    for i in range(n):
        x1 = np.array([pts1_new[:, i]]).reshape((3, 1))
        x2 = np.array([pts2_new[:, i]]).reshape((3, 1))
        r = x2 @ x1.T
        r = np.reshape(r, (1, 9))
        A[i] = r

    #Finding svd of A
    svd_u, svd_sigma, vh = np.linalg.svd(A)
    v = vh.T
    F_1 = v[:, -1]
    F_1 = np.reshape(F_1, (3, 3))

    #Finding svd for F_1 and imposing rank2 (by making last eigenvalue 0)
    svd_u, svd_sigma, vh = np.linalg.svd(F_1)
    svd_sigma[2] = 0
    F_2 = svd_u @ np.diag(svd_sigma) @ vh

    ##denormalize matrix
    Fundamental_mat = t2.T @ F_2 @ t1

    #8 point algo
    Fundamental_mat = Fundamental_mat * (1 / Fundamental_mat[2, 2])
    return Fundamental_mat


def FindFundamentalMatrixRansac(pts1, pts2, num_trials, threshold):

    pts1 = pts1
    pts2 = pts2
    num_trials = num_trials
    threshold = threshold
    len1 = pts1.shape[0]
    point_pairs = 8

    fundamental_arr = []
    inlier_counter = []

    for i in range(num_trials):
        ##sample 8 pts randomly
        i_s = sorted(np.random.choice(len1, point_pairs, replace=False))
        selected_pts1 = pts1[i_s]
        selected_pts2 = pts2[i_s]
        F = FindFundamentalMatrix(selected_pts1, selected_pts2)
        fundamental_arr.append(F)
        ##inlier count
        test1 = pts1
        test2 = pts2
        test1 = np.hstack((test1, np.ones((test1.shape[0], 1), dtype=np.int32)))
        test2 = np.hstack((test2, np.ones((test2.shape[0], 1), dtype=np.int32)))

        inlier_num = 0
        len2 = test1.shape[0]
        ##calculating loss
        for j in range(len2):
            error = abs(test2[j] @ F @ test1[j])

            if error < threshold:
                inlier_num += 1

        inlier_counter.append(inlier_num)

    best_Fundamental_mat = fundamental_arr[np.argmax(inlier_counter)]
    return best_Fundamental_mat


if __name__ == '__main__':
    #Set parameters
    data_path = './data'
    use_ransac = True

    #Load images
    image1_path = os.path.join(data_path, 'notredam_1.jpg')
    image2_path = os.path.join(data_path, 'notredam2.jpg')
    image1 = np.array(Image.open(image1_path).convert('L'))
    image2 = np.array(Image.open(image2_path).convert('L'))


    #Find matching keypoints
    pts1, pts2 = find_matching_keypoints(image1, image2)

    #Builtin opencv function for comparison
    F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
    print("True 8POINT: ", F_true)


    #todo: FindFundamentalMatrix
    if use_ransac:
        F_true_ransac = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold = 0.01, confidence = 0.01, maxIters = 1000)[0]
        print("True RANSAC: ", F_true_ransac)
        F = FindFundamentalMatrix(pts1, pts2)
        F_ransac = FindFundamentalMatrixRansac(pts1, pts2, num_trials=1000, threshold=0.01)
        print("Calc RANSAC: ", F_ransac)

    else:
        F_norm = FindFundamentalMatrix(pts1, pts2)
        print("Calc FundaM:  ", F_norm)




    # Find epilines corresponding to points in second image,  and draw the lines on first image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F_ransac)
    lines1 = lines1.reshape(-1, 3)
    img1, img2 = drawlines(image1, image2, lines1, pts1, pts2)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()


    # Find epilines corresponding to points in first image, and draw the lines on second image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_ransac)
    lines2 = lines2.reshape(-1, 3)
    img1, img2 = drawlines(image2, image1, lines2, pts2, pts1)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()





