images = [" ".join(i.split("-")[2:5]).replace("_"," ").replace("(* ", "BC to ").strip() for i in open("data/links.txt", "r").read().split("\n") if "x-y-z" not in i]
w = open("labels.txt", "w")
w.write("\n".join(images))
w.close()