import numpy as np
tags = list(np.unique([i.split("-")[-1].replace(".jpg", "") for i in open("data/links.txt", "r").read().split("\n") if "x-y-z" not in i]))
w = open("tags.txt", "w")
w.write("\n".join(tags))
w.close()