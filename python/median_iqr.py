import numpy as np
filename = "aucs/3_normed_50var_scaling05_weak_crossent_elu_nodp10"
data = np.load(filename + "_aucs.npy")
for i in range(1, 3):
   data = np.concatenate((data, np.load( filename + "-" +  str(i) + "_aucs.npy")))
#old_results = [0.798607311645, 0.79206946563, 0.788833737591, 0.791421069741, 0.797334900353, 0.786063239229, 0.796501712885, 0.792457677978, 0.790566002351] 
#data = np.concatenate((data, old_results))
data = data[np.where(data < 0.95)]
print(len(data))
median = np.median(data)
q75, q25 = np.percentile(data, [75, 25])
iqr = q75 - q25
print("median : ", median, " iqr : ", iqr)
