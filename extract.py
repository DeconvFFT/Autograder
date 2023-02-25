from PIL import Image as im
import numpy as np
from PIL import ImageDraw,ImageOps, ImageFilter
import sys
if __name__ == "__main__":
    form_im=im.open(sys.argv[1])
    gray_im=ImageOps.grayscale(form_im)
    arr_gray=np.array(gray_im)
    index=0
    for i in range(len(arr_gray)):
        if sum(iter < 60 for iter in arr_gray[i])>8:
            index=i
            print(index)
            break
    # occ=arr_gray[index].find('')
    barcode_rows=np.mean(arr_gray[index:40+index,:],axis=0)
    first=np.zeros(1700)
    for i in range(1700):
        if barcode_rows[i]>60:
            first[i]=255
        else:
            first[i]=0
    a_resh=first.reshape(-1,5).astype('int')
    zero_row=np.zeros(5)
    val_ind=0
    for i in range(len(a_resh)-1):
        if (np.array_equal(a_resh[i],zero_row) and np.array_equal(a_resh[i+1],zero_row)):
            val_ind=i+2
            print(val_ind)
            break
    index={0:'A',1:'B',2:'C',3:'D',4:'E'}
    with open(sys.argv[2],'w+') as f:
        ctr=1
        for j in range(val_ind,85+val_ind):
            f.write(str(ctr)+" ")
            for k in range(5):
                if(a_resh[j][k]==255):
                    f.write(index[k]),
            f.write("\n")
            ctr=ctr+1


