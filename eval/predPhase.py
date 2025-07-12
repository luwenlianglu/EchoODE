import os
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
from scipy.signal import savgol_filter
import scipy
def predphase():
    predMaskVidRoot=r'I:\echo_ode\outputs\convgru_ode\demoVideo\images'
    thre = 10
    method=predMaskVidRoot.split('\\')[-3]
    timesteps = int(method[method.find("timesteps") + len("timesteps")]) if method.find("timesteps")>=0 else 1
    dilation = int(method[method.find("dilation") + len("dilation")]) if method.find("dilation")>=0 else 1
    splitRoot=r"I:\lwl\lvh-seg\data\videos\split\test.txt"
    with open(splitRoot) as f:
        lines=f.readlines()
    dic={}
    for line in lines:
        infos=line.split('|')
        name=infos[0]
        framenum=int(infos[-1])
        phase=[int(infos[x]) for x in range(1,len(infos)-2)]
        dic[name]={"framenum":framenum, "phase":phase}

    #由于原图是从1开始编号，预测的是从0开始编号，所以需要错开一位，即split上编号从1开始，读取mask从0开始
    phasediff=[]
    diffd = []
    diffs = []
    for name in dic.keys():
        maskphase=dic[name]["phase"]
        imagesroot=os.path.join(predMaskVidRoot, name)
        framenum = len(os.listdir(imagesroot))
        maskArea=[]
        for i in range(framenum):
            mask=Image.open(os.path.join(imagesroot, str(i).zfill(3)+'.png'))
            maskarray=np.array(mask)
            pixnum=np.count_nonzero(maskarray==4)
            maskArea.append(pixnum)
        #进行滤波
        maskArea=np.array(maskArea)
        filtered_A=savgol_filter(maskArea, window_length=13,polyorder=3)
        diffA=max(filtered_A)-min(filtered_A)
        d=18
        b=0.5*diffA
        Dn,_=scipy.signal.find_peaks(x=filtered_A, distance=d,prominence=b)
        Sn,_=scipy.signal.find_peaks(x=-filtered_A, distance=d,prominence=b)
        phase_true=dic[name]["phase"]

        for t in phase_true:
            for d in Dn:
                real=idx2srcidx(d,method,timesteps,dilation)
                if abs(t-1-real)<thre:
                    phasediff.append(abs(t-1-real))
                    diffd.append(abs(t-1-real))
            for s in Sn:
                real = idx2srcidx(s, method, timesteps, dilation)
                if abs(t-1-real)<thre:
                    phasediff.append(abs(t-1-real))
                    diffs.append(abs(t-1-real))
        plot=False
        if(plot):
            plt.plot(range(len(maskArea)), maskArea,'b--')
            plt.plot(range(len(filtered_A)), filtered_A, 'r--')
            plt.plot(Dn, filtered_A[Dn], 'xg')
            plt.plot(Sn, filtered_A[Sn], '*g')
            plt.plot(phase_true, filtered_A[phase_true], 'or')
            plt.show()
    print("D: ",np.mean(np.array(diffd)), " S:",np.mean(np.array(diffs))," overall:", np.mean(np.array(phasediff)))
    print(phasediff)
def idx2srcidx(idx, method, timesteps, dilation):
    return idx
    if "convlstm" in method:
        realdur=(timesteps-1)*dilation+1
        return idx//timesteps*realdur+idx%timesteps*dilation
    else:
        return idx

if __name__=="__main__":
    predphase()






