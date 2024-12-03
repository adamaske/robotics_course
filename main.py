import numpy as np

from scipy.linalg import expm
import random

# R is a 2x2 or 3x3 numpy array
# e is epsilon, default value is 1e-13. 
# If no argument is given then default value is used
def is_rotation(R, e=1e-13):
    shape = R.shape
    print("r_shape : ", shape)
    assert( (shape == (2,2)) or (shape == (3,3)))
    
    r_t = np.transpose(R)
    i = np.identity(shape[0])
    r_rt = np.matmul(r_t, R)
    r_i = np.subtract(r_rt, i)
    
    cond_m = np.allclose(r_i, np.zeros(shape=shape), atol=e)
    print("R_t*R -I < e", cond_m)
    
    det = np.linalg.det(R)
    cond_d = np.isclose(det, 1, atol=e)
    
    print("det(R) : ", det)
    print("det(R) = 1 +- e : ", cond_d)
    
    if (cond_m and cond_d):
        return True
    else:
        return False

def skew(x):
    assert(len(x) == 3)
    S = np.array((( 0, -x[2], x[1]),
                  ( x[2], 0, -x[0]),
                  ( -x[1], x[0], 0)))
    
    return S

# w is unit vector
# theta is angle in radians
def rodrigues(w,theta):
    I = np.identity(3)
    w_ = skew(w) # [w]
    w_2 = np.matmul(w_, w_)
    
    s_t_w_ = np.sin(theta) * w_
    #print("sin(t)*[w] : ", s_t_w_)
    
    c_t = (1 - np.cos(theta))
    c_t_w_2 = c_t * w_2
    #print("(1 - cos(t))[w]^2 : ", c_t_w_2)
    
    i_s = np.add(I, s_t_w_) # I + sin(t)[w]
    R = np.add(i_s, c_t_w_2)
    return R

def verify_rodrigues(w, theta):
    R1 = rodrigues(w, theta)
    
    w_ = skew(w)
    w_t = np.multiply(theta, w_) 
    R2 = expm(w_t)

    r1_r2 = np.subtract(R1, R2)
    print("R1 - R2 : ")
    print(r1_r2)
    
    cond = np.allclose(r1_r2, np.zeros(r1_r2.shape[0]), atol=1e-13)
    print("R1 - R2 = 0 :", cond)
    return cond, R1, R2
    



def rotation_matrix_logarithm(R):
    r = np.array(R)
    #case a
    if np.allclose(R, np.identity(R.shape[0]), atol=1e-13):
        print("theta = 0, w is undefined")
        return ValueError #theta = 0, w is undefined
    
    #case b
    #if trace of R = -1 then theta = pi, 
    tr_r = r.trace()
    print("tr_r : ", tr_r)
    if np.isclose(tr_r, -1):
        t = np.pi
        v1 = np.array(((r[0][2], r[1][2], 1 + r[2][2])))
        v2 = np.array(((r[0][1], 1+ r[1][1], r[2][1])))
        v3 = np.array(((1 + r[0][0], r[1][0], 1 + r[2][0])))
        
        v1_ = 1.0 / np.sqrt(2*(1+r[2][2]))
        v2_ = 1.0 / np.sqrt(2*(1+r[1][1]))
        v3_ = 1.0 / np.sqrt(2*(1+r[0][0]))
        
        w = v1_ * v1
        #w = v2_ * v2
        #w = v3_ * v3
        return t, w
    
    #case c
    t = np.arccos(0.5 * (tr_r - 1))
    w = np.multiply(np.divide(1.0, 2*np.sin(t)), np.subtract(r, r.transpose())) 
    return t, w

x = np.random.randn(3)
w = np.divide(x, np.sqrt(x.dot(x)))
t = random.uniform(-2*np.pi, 2*np.pi)
success, r_rod, r_exp = verify_rodrigues(w, t)

t_o, w_o = rotation_matrix_logarithm(r_rod)
if t is ValueError:
    print("t_o : 0")
    print("w_o : undefined")
    exit()
    
w_o_unskewed= np.array(((w_o[2][1], w_o[0][2], w_o[1][0]))) # for w, -w is also valid solution
print("t input : ", t)
print("w input : ", w)

print("t output : ", t_o)
print("w output : ", w_o_unskewed)

t_diff = np.isclose(np.abs(t), np.abs(t_o), atol=1e-13)
w_diff = np.allclose(np.abs(w), np.abs(w_o_unskewed), atol=1e-13)
print("t close: ", t_diff)
print("w close: ", w_diff)
