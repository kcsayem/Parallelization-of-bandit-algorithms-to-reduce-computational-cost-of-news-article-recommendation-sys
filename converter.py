import os


import os

rootdir = "/home/elkhan/BanditLib"
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.endswith(".py"):
            os.system (f"2to3  -w -n {os.path.join(rootdir,subdir,file)} ")
            os.system(f"reindent -n {os.path.join(rootdir,subdir,file)}")
