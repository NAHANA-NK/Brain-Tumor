import os



dataset=os.listdir(r"D:\Dataset\archive\Training")

print(dataset)
j=0
for i in dataset:
    files=os.listdir(r"D:\Dataset\archive\Training\\"+i)
    for f in files:
        path=os.path.join(r"D:\Dataset\archive\Training\\"+i,f)

    j=j+1


