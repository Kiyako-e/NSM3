import numpy as np

def owa(x, y):
    C5 = []
    diff = y - x
    for i in range(len(x)):
        for j in range(i, len(y)):
            C5.append((diff[i] + diff[j]) / 2)

    owa = sorted(C5)
    h_l = np.median(owa)
    return {"owa": owa, "h.l": h_l}
