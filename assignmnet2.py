
from scipy.linalg import expm
import numpy as np
J_s= np.array(((1, 0, 1, 0, 1, 1, 1),
               (0, 0, 0, 0, 0, 0, 0),
               (0, 1, 0, 1, 0, 0, 0),
               (0, 0, 0, 0, 0, 0, 0),
               (0, 0.34, 0, 0.74, 0, 1.14, 0),
               (0, 0, 0, 0, 0, 0, 0)))
J_b= np.array(((1, 0, 1, 0, 1, 0, 1),
               (0, 0, 0, 0, 0, 0, 0),
               (0, 1, 0, 1, 0, 1, 0),
               (0, 0, 0, 0, 0, 0, 0),
               (0, -0.95, 0, -0.55, 0, -0.15, 0),
               (0, 0, 0, 0, 0, 0, 0)))

F_s = np.array(((1, 1, 1, 1, 1, 1)))
t_s = np.matmul(J_s.transpose(), F_s)

F_b = np.array(((1, 1, 1, 1, 1, 1)))
t_b = np.matmul(J_b.transpose(), F_b)
print("t_s: ", t_s)
print("t_b: ", t_b)
J_sw = J_s[0:3]
J_sv = J_s[3:]
J_bw = J_b[0:3]
J_bv = J_b[3:]

a_sw = np.matmul(J_sw, np.transpose(J_sw))
a_sv = np.matmul(J_sv, np.transpose(J_sv))
a_bw = np.matmul(J_bw, np.transpose(J_bw))
a_bv = np.matmul(J_bv, np.transpose(J_bv))
#values, vectors
sw_w, sw_v = np.linalg.eig(a_sw)
sv_w, sv_v = np.linalg.eig(a_sv)

bw_w, bw_v = np.linalg.eig(a_bw)
bv_w, bv_v = np.linalg.eig(a_bv)

su2_w = np.max(sw_w) / np.min(sw_w[sw_w != 0])
su2_v = np.max(sw_v) / np.min(sw_v[sw_v != 0])
bu2_w = np.max(bw_w) / np.min(bw_w[bw_w != 0])
bu2_v = np.max(bw_v) / np.min(bw_v[bw_v != 0])
print("su2_w : ", su2_w)
print("su2_v : ", su2_v)
print("bu2_w : ", bu2_w)
print("bu2_v : ", bu2_v)

exit() 

def make_3x1_position_vector(x, y, z):
    return np.array(((x, y, z)))

def make_3x3_rotation_matrix(axis:str, theta):
    t_rad = np.deg2rad(theta)
    ct = np.cos(t_rad)
    st = np.sin(t_rad)
    
    R = np.identity(3)
     
    if axis == 'x':
        R[1][1] = ct
        R[2][1] = st
        
        R[1][2] = -st
        R[2][2] = ct
        return R
    if axis == 'y':
        R[0][0] = ct
        R[0][2] = st
        
        R[2][0] = -st
        R[2][2] = ct
        return R
    if axis == 'z':
        R[0][0] = ct
        R[1][0] = st
        
        R[0][1] = -st
        R[1][1] = ct
        return R
    return R
    
def make_4x4_transformation_matrix(r, p):
    assert(r.shape == (3,3))
    assert(p.shape == (3,))
    
    T = np.array(((r[0][0], r[0][1], r[0][2], p[0]), 
                  (r[1][0], r[1][1], r[1][2], p[1]), 
                  (r[2][0], r[2][1], r[2][2], p[2]), 
                  (0, 0, 0, 1)))
    
    return T
s_r = make_3x3_rotation_matrix('x', 90)
s_p = make_3x1_position_vector(0,0,0)
s = make_4x4_transformation_matrix(make_3x3_rotation_matrix('x', 0), make_3x1_position_vector(0,0,2))

#print("y axis 90 degrees rotation", make_3x3_rotation_matrix('y', 90))

def blocked_screw(screw):
    assert(screw.shape == (6,))
    w = np.array(screw[0:3])
    v = np.array(screw[3:6])
    S_B = np.array(((      0,   -w[2],   w[1], v[0]),
                    (   w[2],       0,  -w[0], v[1]),
                    (  -w[1],    w[0],      0, v[2]),
                    (      0,       0,      0,    0)))
    
    print("screw :", screw)
    print("w :", w)
    print("v :", v)
    print("blocked_screw :")
    print(S_B)
    return S_B
    
    
w1, v1 = (0, 0, 1), (0, 0, 0)
S1 = np.append(w1, v1)

w2, v2= (1, 0, 0), (0, -70.42, 0)
S2 =  np.append(w2, v2)

w3, v3 = (1, 0, 0), (0, -180.8,0)
S3 =  np.append(w3, v3)

w4, v4 = (1, 0, 0), (0, -276.8, 0)
S4 =  np.append(w4, v4)

w5, v5 = (0, 0, 1), (0, 66.39, 0)
S5 =  np.append(w5, v5)

w6, v6 = (1,0,0), (0,-350,0)
S6 = np.append(w6, v6)

M = np.array((( 1, 0, 0, 100),
              ( 0, 1, 0, 0),
              ( 0, 0, 1, 350),
              ( 0, 0, 0, 1)))

s1b = blocked_screw(S1).transpose()
s2b = blocked_screw(S2).transpose()
s3b = blocked_screw(S3).transpose()
s4b = blocked_screw(S4).transpose()
s5b = blocked_screw(S5).transpose()
s6b = blocked_screw(S6).transpose()
s_array = [S1, S2, S3, S4, S5, S6]
t_array = [0, 0, 0, 0, 0, 0]

t1, t2, t3, t4, t5, t6 = 0, 0, 0, 0, 0, 41

e1 = np.array(expm(s1b * t1))
e2 = np.array(expm(s2b * t2))
e3 = np.array(expm(s3b * t3))
e4 = np.array(expm(s4b * t4))
e5 = np.array(expm(s5b * t5))
e6 = np.array(expm(s6b * t6))

e12 = np.matmul(e1, e2)
e123 = np.matmul(e12, e3)
e1234 = np.matmul(e123, e4)
e12345 = np.matmul(e1234, e5)
e123456 = np.matmul(e12345, e6)
T = np.matmul(e123456, M)

print()
print("T = ")
print(T)