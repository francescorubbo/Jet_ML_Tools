import numpy as np
filename = "aucs/3_normed_50var_scaling05_weak_fixedsoftmax_relu_nodp"
data = np.load(filename + "_aucs.npy")
for i in range(1, 4):
   data = np.concatenate((data, np.load( filename + str(i) + "_aucs.npy")))
#old_results = [0.771145197776, 0.771518959455,  0.761938593935,  0.773811097392, 0.762632043267,  0.773105506438, 0.766450771106, 0.773154072926, ] 
#data = np.concatenate((data, old_results))
data = data[np.where(data < 0.95)]
print(len(data))
median = np.median(data)
q75, q25 = np.percentile(data, [75, 25])
iqr = q75 - q25
print("median : ", median, " iqr : ", iqr)
