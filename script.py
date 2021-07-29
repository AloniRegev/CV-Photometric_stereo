import math
import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# method to find an Sum Squer Distances bitween 2 paches
def findSSD(patchLeft, patchRight):
    return np.sum((np.array(patchLeft, dtype=np.float32) - np.array(patchRight, dtype=np.float32)) ** 2)  # SSD formula


# method to find an Normalized Cross Correlation bitween 2 paches
def findNCC(patchLeft, patchRight):
    AB = np.sum(np.array(patchLeft, dtype=np.float32) * np.array(patchRight, dtype=np.float32))  # Numerator
    # Denominator calculation
    A = math.sqrt(np.sum(np.array(patchLeft, dtype=np.float32) ** 2))
    B = math.sqrt(np.sum(np.array(patchRight, dtype=np.float32) ** 2))

    return (AB / (A * B))  # NCC formula


# test mathod to calculet the  mean of absolute differences in pixels compare to ground truth disparity map
def findAvgErr(groundTrout, dImg):
    return np.mean(np.abs(np.array(groundTrout, dtype=np.float32) - np.array(dImg,
                                                                             dtype=np.float32)))  # Finding the absolute difference in the average value calculation


# test mathod to calculet the  median of absolute differences in pixels compare to ground truth disparity map
def findMedErr(groundTrout, dImg):
    return np.median(np.abs(np.array(groundTrout, dtype=np.float32) - np.array(dImg,
                                                                               dtype=np.float32)))  # Finding the absolute difference in the median alue calculation


# test mathod to calculet the percentage of disparities whose error is above 0.5 compare to ground truth disparity map
def findBad05(groundTrout, dImg):
    arrayRange = np.abs(np.array(groundTrout, dtype=np.float32) - np.array(dImg, dtype=np.float32))
    outOfRange = np.count_nonzero(arrayRange > 0.5)  # count the disparity values that grater then 0.5

    return (outOfRange / arrayRange.size) * 100


# test mathod to calculet the percentage of disparities whose error is above 4 compare to ground truth disparity map
def findBad4(groundTrout, dImg):
    arrayRange = np.abs(np.array(groundTrout, dtype=np.float32) - np.array(dImg, dtype=np.float32))
    outOfRange = np.count_nonzero(arrayRange > 4)  # count the disparity values that grater then 4

    return (outOfRange / arrayRange.size) * 100


def run_script(imgLeft, imgRight, groundTrout, patch, method=findSSD):
    rows, cols = imgLeft.shape[:2]
    offset = math.floor(patch / 2)
    dImg = np.zeros(imgLeft.shape, dtype=np.uint8)  # init new empthy image withe the same dimantions of the left image
    for r in range(rows):  # move over every epipol line
        print("Percentage complete: ", round(r / rows * 100, 2))  # pricent to complition

        # todo remove
        # if r % 15 == 0:
        #     cv.imshow("Test", dImg)
        #     cv.waitKey(0)

        # up and down bounds of the patch
        upperBound = max(0, r - offset)
        lowerBound = min(rows, r + offset + 1)

        xl = 0
        for c in range(cols):
            if c < patch / 2:
                pass
            else:
                # left and right bounds of the left patch
                leftBound = max(0, c - offset)
                rightBound = min(cols, c + offset + 1)

                leftPatch = imgLeft[upperBound:lowerBound, leftBound:rightBound]  # calculate the right patch
                patchH, patchW = leftPatch.shape[:2]  # patch dimensions

                bestError = -1

                for cRight in range(c - offset):
                    # for cRight in range(cols - (patchW + 1)):
                    rightPatch = imgRight[upperBound:lowerBound, cRight:cRight + patchW]  # calculate the right patch

                    if leftPatch.shape != rightPatch.shape:
                        print("Error occurred in patches")
                        return
                    # call for a method to match between the patches (SSD, NCC)
                    error = method(leftPatch, rightPatch)

                    # in a case of SSD call get the minimum result
                    if method == findSSD and (bestError == -1 or error < bestError):
                        bestError = error

                        # if error > 4:
                        #     xl = c
                        #
                        # else:
                        xl = cRight + offset

                        # todo remove
                        # leftPatchRes = leftPatch
                        # rightPatchRes = rightPatch

                    # in a case of NCC call get the maximum result
                    elif method == findNCC and (bestError == -1 or error > bestError):
                        bestError = error

                        # if error<0.92:
                        #     xl = c
                        # else:
                        xl = cRight + offset

                        # todo remove
                        # leftPatchRes=leftPatch
                        # rightPatchRes=rightPatch

                dImg[r, c] = c - xl
                # todo remove
                # print("bestError: ", bestError)

                # todo remove
                # if c == 240 and ((r%10)==0):
                #     cv.imshow("leftPatch", leftPatchRes)
                #     cv.imshow("rightPatch", rightPatchRes)
                #     cv.waitKey(0)

    # todo: compute the error of the disparity map from the ground truth

    avg = findAvgErr(groundTrout, dImg)
    med = findMedErr(groundTrout, dImg)
    five = findBad05(groundTrout, dImg)
    four = findBad4(groundTrout, dImg)

    return dImg, avg, med, five, four

    # todo remove
    # print(leftPatch.shape, rightPatch.shape)
    # Take all patches on row r that have size same as leftPatch


if __name__ == "__main__":

    ks = [3, 9, 15]
    problems = {
        # "Moebius": ("./inputs/Moebius/view1.png","./inputs/Moebius/view5.png","./inputs/Moebius/disp1.png"),
        # "Art":("./inputs/Art/im_left.png","./inputs/Art/im_right.png","./inputs/Art/disp_left.png"),
        "Dolls": ("./inputs/Dolls/im_left.png", "./inputs/Dolls/im_right.png", "./inputs/Dolls/disp_left.png")}
    methods = {"SSD": findSSD, "NCC": findNCC}

    outputdir = "./Q2output/"
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    for p in problems:
        img1Path, img2Path, img3Path = problems[p]
        for k in ks:
            for m in methods:
                outputfile = outputdir + f"{p}-{str(k)}-{m}.txt"
                if os.path.exists(outputfile):
                    continue
                print(outputfile)
                method = methods[m]
                imgLeft = cv.imread(img1Path, 0)  # queryimage # left image
                imgRight = cv.imread(img2Path, 0)  # trainimage # right image
                groundTrout = cv.imread(img3Path, 0)
                groundTrout = np.array(groundTrout)/3
                dImg, avg, med, five, four = run_script(imgLeft, imgRight, groundTrout, k, method)

                outputfile = outputdir + f"{p}-{str(k)}-{m}.txt"
                file = open(outputfile, "w")
                file.write(f"{avg}\t{med}\t{five}\t{four}")
                file.close()
                outputimage = outputdir + f"{p}-{int(k)}-{m}.png"
                cv.imwrite(outputimage, np.array(dImg) * 3)



