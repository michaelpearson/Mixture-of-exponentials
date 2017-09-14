import EM
import Generate_Data as data
import matplotlib.pyplot as plt


k = 5
d = 2

data.generate_data(k, d)
EM.run_em(k)

plt.show()
