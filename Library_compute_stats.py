from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

#Ho usato cnvenzione opaper sensors andando a calcolare le statistiche sulla classde grossa
def compute_descriptors(Y,P):
    if(Y.shape[1]>1):
        Yp = np.argmax(Y, axis=1)
        Pp = np.argmax(P, axis=1)
        Yn = np.argmin(Y, axis=1)
        Pn = np.argmin(P, axis=1)
    else:
        Yp = Y
        Pp = P
        Yn = -Y
        Pn = -P

    cm_p = confusion_matrix(Yp,Pp)
    accuracy_p = accuracy_score(Yp,Pp)
    prec_p = precision_score(Yp,Pp)
    rec_p = recall_score(Yp,Pp)
    f1_p = f1_score(Yp,Pp)

    cm_n = confusion_matrix(Yn,Pn)
    accuracy_n = accuracy_score(Yn,Pn)
    prec_n = precision_score(Yn,Pn)
    rec_n = recall_score(Yn,Pn)
    f1_n = f1_score(Yn,Pn)

    return cm_p,accuracy_p,prec_p,rec_p,f1_p, cm_n,accuracy_n,prec_n,rec_n,f1_n

def compute_descriptors_bal(Y,P):
    cm = confusion_matrix(Y,P)
    accuracy = accuracy_score(Y,P)
    prec = precision_score(Y,P, average="weighted")
    rec = recall_score(Y,P, average="weighted")
    f1 = f1_score(Y,P, average="weighted")
    return cm,accuracy,prec,rec,f1

def avg_fold_res(list_res,out_file_name):
    cm_p = np.zeros((2,2))
    accuracy_p = 0
    prec_p = 0
    rec_p = 0
    f1_p = 0
    cm_n = np.zeros((2,2))
    accuracy_n = 0
    prec_n = 0
    rec_n = 0
    f1_n = 0
    for i in range(len(list_res)):
        cm_p = cm_p + list_res[i][0]
        accuracy_p = accuracy_p + list_res[i][1]
        prec_p = prec_p + list_res[i][2]
        rec_p = rec_p + list_res[i][3]
        f1_p = f1_p + list_res[i][4]
        cm_n  = cm_n  + list_res[i][5]
        accuracy_n  = accuracy_n  + list_res[i][6]
        prec_n  = prec_n  + list_res[i][7]
        rec_n  = rec_n + list_res[i][8]
        f1_n  = f1_n  + list_res[i][9]

    accuracy_p = accuracy_p/len(list_res)
    prec_p = prec_p/len(list_res)
    rec_p = rec_p/len(list_res)
    f1_p = f1_p/len(list_res)

    accuracy_n = accuracy_n/len(list_res)
    prec_n = prec_n/len(list_res)
    rec_n = rec_n/len(list_res)
    f1_n = f1_n/len(list_res)

    with open(out_file_name, 'a') as the_file:
        the_file.write(str(accuracy_p)+ " ")
        the_file.write(str(prec_p)+ " ")
        the_file.write(str(rec_p)+ " ")
        the_file.write(str(f1_p)+ "\n")

    return cm_p,accuracy_p,prec_p,rec_p,f1_p,cm_n,accuracy_n,prec_n,rec_n,f1_n


