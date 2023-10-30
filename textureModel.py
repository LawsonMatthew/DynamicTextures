from matplotlib import pyplot as plt
import argparse
import numpy as np
from scipy.linalg import pinv, svd

def flatten_video(video_data):
    Y = []
    for frame in M:
        Y.append(frame.flatten())
    Y = np.array(Y)
    return Y


def rmse(predicted_frame, real_frame):
    # Ensure both frames have the same dimensions
    if predicted_frame.shape != real_frame.shape:
        raise ValueError("Both frames must have the same dimensions.")

    # Calculate the squared differences element-wise
    squared_diff = (predicted_frame - real_frame) ** 2

    # Calculate the mean squared difference
    mean_squared_diff = np.mean(squared_diff)

    # Calculate RMSE by taking the square root of the mean squared difference
    rmse_value = np.sqrt(mean_squared_diff)

    return rmse_value

input_file = "dt1_train.npy"
q = 10
output_file = "output"


#Load in the dynamic texture data.
M = np.load(input_file)
M = M[:-1,:,:]
real_frame = M[-1,:,:]
frames, height, width = M.shape

#Flatten each frame in the dynamic texture to create the Y matrix.
Y = flatten_video(M)
Y = Y.T

#Use scipy svd to find U, sigma, and V.
U, sigma, V = svd(Y)
U_hat = U[:,:q]
sigma_q = sigma[:q]
sigma_hat = np.diag((sigma_q))
V_hat = V[:q,:]

# C transition matrix is equal to U_hat
C = U_hat

# # Compute X,state space, the product of sigma_hat and V_hat
X = sigma_hat @ V_hat

#Use C and X to calculate A, the transition matrix. X1_f_1 is matrix of X1-Xf-1 stacked as rows
X1_f_1 = X[:,:-1]
X2_f = X[:,1:]
A = X2_f @ pinv(X1_f_1)

#Calculate noise term modeled by normal distribution
prediction_errors = np.zeros((q, frames - 1))
for t in range(frames - 1):
    pt = X2_f[:, t] - np.dot(A, X1_f_1[:, t])
    prediction_errors[:, t] = pt

# Compute the covariance matrix Q using the specified equation
Q = (1 / (frames - 1)) * (prediction_errors @ prediction_errors.T)

# Perform SVD on Q to calculate W
U_w, Sigma_w, _ = svd(Q, full_matrices=False)
W = U_w @ np.diag(np.sqrt(Sigma_w))
vt = np.random.normal(0, 1, (q))

# Calculate ~xt+1 using the state transition equation with noise term
Xt_next_noise = A @ X[:,-1] + W @ vt
#Convert using appearance model
Yt_next_noise = C @ Xt_next_noise

# Calculate ~xt+1 using the state transition equation without noise term
Xt_next = A @ X[:,-1] 
#Convert using appearance model
Yt_next = C @ Xt_next

# #Reshape the array to correspond with image dimensions and save
new_frame_noise = Yt_next_noise.reshape((height,width))
new_frame = Yt_next.reshape((height,width))

#Save to .npy files
np.save(output_file,new_frame)    
np.save("outputNoise.npy",new_frame_noise)
### FINISH ME
