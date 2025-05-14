import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


def drawQQplot(data: np.array, id: int) -> tuple:
	# Set of points - real data
	y, N = np.sort(data), len(data)
	mean, std = np.mean(y), np.std(y)
	ppf = norm(loc=mean, scale=std).ppf
	x = [ppf(i / (N + 2)) for i in range(1, N + 1)]
	
	plt.scatter(x, y)

	# Ideal line - normal distribution
	dmin, dmax = np.min([x, y]), np.max([x, y])
	diag = np.linspace(dmin, dmax, N)
	plt.plot(diag, diag, color="red", linestyle="--")
	plt.gca().set_aspect("equal")

	plt.xlabel("Normal quantiles")
	plt.ylabel("Data quantiles")

	plt.savefig("qq-plots/" + str(id) + ".png")

	return mean, std


## Shapiro analytic test
## stat, p = shapiro(data[i])
## print(stat, p, "TRUE" if p > 0.05 else "FALSE")

if __name__ == "__main__":
	with open("distr_gauss") as f:
		data = np.array(list(map(lambda x: list(map(int, x.split())), f.read().split("\n")))[:-1], dtype=int)
	for i in range(0, len(data), 10):
		print(*drawQQplot(data[i], i))
