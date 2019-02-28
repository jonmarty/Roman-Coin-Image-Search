import os
os.system("python setup/retrieve_links.py")
os.system("python setup/retrieve_images.py")
os.system("python setup/generate_tags.py")
os.system("python setup/generate_labels.py")
