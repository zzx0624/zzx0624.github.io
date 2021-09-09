import csv
import time
from numpy.lib.npyio import savez_compressed
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
# itemid = []
# with open('ml-1m.train.rating', "r") as f:
#     line = f.readline()
#     while line != None and line != "":
#         arr = line.split("\t")
#         itemid.append(int(arr[1]))
#         line = f.readline()
# with open('ml-1m.test.rating', "r") as f:
#     line = f.readline()
#     while line != None and line != "":
#         arr = line.split("\t")
#         itemid.append(int(arr[1]))
#         line = f.readline()
# with open('ml-1m.test.negative', "r") as f:
#     line = f.readline()
#     while line != None and line != "":
#         arr = line.split("\t")
#         for i in range(1, len(arr)):
#             itemid.append(int(arr[i]))
#         line = f.readline()
# itemid = list(set(itemid))
# print(len(itemid))
# vedio = []
# audio = []
# C
# print(vedio)
# with open("track2_audio_feature.csv", 'r', encoding='utf-8-sig') as f:
#     reader = csv.reader(f)
#     for low in reader:
        
#         if low[128] != 'item_id' and low[128] != '':
#             audio.append(low[128])
# audio = np.array(audio)
# print(audio)
# x = 0
# a = []
# b = []
# for i in range(len(vedio)):
#     for j in range(len(audio)):
#         if int(vedio[i]) == int(float(audio[j])) and x < 3708:
#             print(0)
#             x = x + 1
#             a.append(vedio[i])
#             b.append(audio[j])
# a = np.array(a)
# b = np.array(b)
# print(len(a))
# print(len(b))

# y = np.array(a)
# df = pd.DataFrame(y)
# df.to_csv("a.csv", encoding="utf-8-sig", header=False, index=False)
# y = np.array(b)
# df = pd.DataFrame(y)
# df.to_csv("b.csv", encoding="utf-8-sig", header=False, index=False)


item = []
with open("item.csv", 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for low in reader:
            item.append(int(low[0]))
vedio = []
with open("last.csv", 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for low in reader:
        if low[0] != 'item_id' and low[0] != '':
            if int(low[0]) in item:
                vedio.append(low)
vedio = np.array(vedio)
print(len(vedio))
audio =[]
with open("track2_audio_feature.csv", 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for low in reader:
        if low[128] != 'item_id' and low[128] != '':
            if int(float(low[128])) in item:
                audio.append(low)
audio = np.array(audio)
y = np.array(vedio)
df = pd.DataFrame(y)
df.to_csv("a.csv", encoding="utf-8-sig", header=False, index=False)
y = np.array(audio)
df = pd.DataFrame(y)
df.to_csv("b.csv", encoding="utf-8-sig", header=False, index=False)