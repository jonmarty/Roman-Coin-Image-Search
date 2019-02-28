import os
images = [i for i in open("data/links.txt", "r").read().split("\n") if "x-y-z" not in i]
if not os.path.exists("./images"):
    os.mkdir("images")
os.chdir("images")
for image in images:
    os.system("curl -k -o \"{}\" \"{}\"".format("-".join(image.split("-")[2:]), image.replace("d3k6u6bv48g1ck.cloudfront.net","www.coinshome.net")))
os.chdir("..")
