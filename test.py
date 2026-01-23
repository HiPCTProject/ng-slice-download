import numpy as np


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

q_y_90 = np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2, 0])
q = np.array([0.5823236107826233, -0.37698644399642944, -0.3739404082298279, 0.615588366985321])

q_rotated = quat_multiply(q_y_90, q)
print(q_rotated)
