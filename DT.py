
import os

import numpy as np
from skimage import io
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics import classification_report

def glcm_feat(path):
    x = io.imread(path)

    if(len(x.shape)==3):
        x= x[:, : ,2]
    nir = x[:, : ]
    glcm = greycomatrix(x, [1], [np.pi/2], levels=256, normed=True, symmetric=True)

    li=[];
    # contrast’, ‘dissimilarity’, ‘homogeneity’, ‘energy’, ‘correlation’, ‘ASM’},
    li.append(greycoprops(glcm,prop='contrast')[0][0])
    li.append(greycoprops(glcm,prop='dissimilarity')[0][0])
    li.append(greycoprops(glcm,prop='homogeneity')[0][0])
    li.append(greycoprops(glcm,prop='energy')[0][0])
    li.append(greycoprops(glcm,prop='correlation')[0][0])
    li.append(greycoprops(glcm,prop='ASM')[0][0])
    return li

fol=os.listdir(r'D:\Dataset\archive\Training')
x=[]
y=[]
j=0
for i in fol:
    files = os.listdir(os.path.join(r'D:\Dataset\archive\Training',i))
    for f in files:
        fn = os.path.join(r'D:\Dataset\archive\Training', i,f)
        print(fn)
        fe=glcm_feat(fn)
        x.append(fe)
        y.append(j)
    j=j+1


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X = x # petal length and width
y = y
X_train, X_test, y_train, y_test = train_test_split(X,y ,
                                   random_state=104,
                                   test_size=0.2)
# DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=2)
tree_clf.fit(X_train, y_train)

res=tree_clf.predict(X_test)
cm = confusion_matrix(y_test,res)

print(cm)
print(classification_report(y_test, res))

