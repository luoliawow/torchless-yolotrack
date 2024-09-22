# Ultralytics YOLO 🚀, AGPL-3.0 license

import copy

import cv2
import numpy as np


class GMC:
    """
    Generalized Motion Compensation (GMC) class for tracking and object detection in video frames.

    This class provides methods for tracking and detecting objects based on several tracking algorithms including ORB,
    SIFT, ECC, and Sparse Optical Flow. It also supports downscaling of frames for computational efficiency.

    Attributes:
        method (str): The method used for tracking. Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'.
        downscale (int): Factor by which to downscale the frames for processing.
        prevFrame (np.ndarray): Stores the previous frame for tracking.
        prevKeyPoints (List): Stores the keypoints from the previous frame.
        prevDescriptors (np.ndarray): Stores the descriptors from the previous frame.
        initializedFirstFrame (bool): Flag to indicate if the first frame has been processed.

    Methods:
        __init__: Initializes a GMC object with the specified method and downscale factor.
        apply: Applies the chosen method to a raw frame and optionally uses provided detections.
        applyEcc: Applies the ECC algorithm to a raw frame.
        applyFeatures: Applies feature-based methods like ORB or SIFT to a raw frame.
        applySparseOptFlow: Applies the Sparse Optical Flow method to a raw frame.
        reset_params: Resets the internal parameters of the GMC object.

    Examples:
        Create a GMC object and apply it to a frame
        >>> gmc = GMC(method="sparseOptFlow", downscale=2)
        >>> frame = np.array([[1, 2, 3], [4, 5, 6]])
        >>> processed_frame = gmc.apply(frame)
        >>> print(processed_frame)
        array([[1, 2, 3],
               [4, 5, 6]])
    """

    def __init__(self, method: str = "sparseOptFlow", downscale: int = 2) -> None:
        """
        Initialize a Generalized Motion Compensation (GMC) object with tracking method and downscale factor.

        Args:
            method (str): The method used for tracking. Options include 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'.
            downscale (int): Downscale factor for processing frames.

        Examples:
            Initialize a GMC object with the 'sparseOptFlow' method and a downscale factor of 2
            >>> gmc = GMC(method="sparseOptFlow", downscale=2)
        """
        super().__init__()

        self.method = method
        self.downscale = max(1, downscale)

        if self.method == "sparseOptFlow":
            self.feature_params = dict(
                maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04
            )

        elif self.method in {"none", "None", None}:
            self.method = None
        else:
            raise ValueError(f"Error: Unknown GMC method:{method}")

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False

    def apply(self, raw_frame: np.array, detections: list = None) -> np.array:
        """
        Apply object detection on a raw frame using the specified method.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).
            detections (List | None): List of detections to be used in the processing.

        Returns:
            (np.ndarray): Processed frame with applied object detection.

        Examples:
            >>> gmc = GMC(method="sparseOptFlow")
            >>> raw_frame = np.random.rand(480, 640, 3)
            >>> processed_frame = gmc.apply(raw_frame)
            >>> print(processed_frame.shape)
            (480, 640, 3)
        """
        if self.method == "sparseOptFlow":
            return self.applySparseOptFlow(raw_frame)
        else:
            return np.eye(2, 3)

    def applySparseOptFlow(self, raw_frame: np.array) -> np.array:
        """
        Apply Sparse Optical Flow method to a raw frame.

        Args:
            raw_frame (np.ndarray): The raw frame to be processed, with shape (H, W, C).

        Returns:
            (np.ndarray): Processed frame with shape (2, 3).

        Examples:
            >>> gmc = GMC()
            >>> result = gmc.applySparseOptFlow(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]))
            >>> print(result)
            [[1. 0. 0.]
             [0. 1. 0.]]
        """
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # Find the keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H

        # Find correspondences
        matchedKeypoints, status, _ = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)

        # Leave good correspondences only
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (prevPoints.shape[0] > 4) and (prevPoints.shape[0] == prevPoints.shape[0]):
            H, _ = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print("WARNING: not enough matching points")

        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        return H

    def reset_params(self) -> None:
        """Reset the internal parameters including previous frame, keypoints, and descriptors."""
        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None
        self.initializedFirstFrame = False
