# This message encapsulates the result of a keypoint detection
std_msgs/Header header

# kp_lm, kp_rm, center, score
float64[2] left_mean
float64[2] right_mean
float64[2] center
float64 score
