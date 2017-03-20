import matplotlib.pyplot as plt
import seaborn as sns
from task1 import mixture, draw_means, draw_weights

K = 5
d = 1.5
lower = 0
upper = 20
m = draw_means(K, d, lower, upper)
w = draw_weights(K)

sample_size = 1000

Q = mixture(w, m)
sample = Q.draw_sample(sample_size)
sns.distplot(sample, hist=True, rug=True)
plt.show()