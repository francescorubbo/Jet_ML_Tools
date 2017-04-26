import numpy as np
filename = "aucs/3_normed_400var_scaling05_weak_sigmoid"
data = np.load(filename + "_aucs.npy")
#for i in range(1, 3):
#   data = np.concatenate((data, np.load( filename + str(i) + "_aucs.npy")))
old_results = [ 0.781313655809,  0.774266870165, 0.776908773847, 0.767938932443, 0.767445943837, 0.771937367686,  0.7805979344,  0.79756290016, 0.799661489596, 0.799137655062,  0.798608818527, 0.795280362145, 0.803590686275, 0.792676070428, 0.784839489215, 0.774504681565,  0.789506672443, 0.782566611876, 0.77529240395,  0.790957812395, 0.770623970171, 0.782146051409]
data = np.concatenate((data, old_results))
data = data[np.where(data < 0.95)]
print(len(data))
median = np.median(data)
q75, q25 = np.percentile(data, [75, 25])
iqr = q75 - q25
print("median : ", median, " iqr : ", iqr)
