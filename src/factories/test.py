import glob
import pathlib


logs = glob.glob("./storage/metrics/*.json")

logs.append("f/f/c.json")
for log in logs:
    print(pathlib.PurePath(log).parts)
