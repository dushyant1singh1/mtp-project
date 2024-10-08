import numpy as np
import matplotlib.pyplot as plt

def plotCDF(psnr, ssim,array_type,acc,k,w,title):
    # sort the data in ascending order 
    psnr = psnr[:40]
    x1 = np.sort(psnr[:40])
    N = len(psnr)
    # get the cdf values of y 
    y1 = np.arange(N) / float(N) 

    ssim = ssim[:40]
    x2 = np.sort(ssim[:40])
    n = len(ssim)
    y2 = np.arange(n)/float(n)
    


    fig , (ax1,ax2) = plt.subplots(1,2,figsize=(8,5))

    #first for psnr
    ax1.plot(x1,y1,color = 'blue')
    ax1.set_title('PSNR CDF after sorting')
    ax1.set_xlabel('Difference in accuracy')
    ax1.set_ylabel('CDF of accuracy differences')

    ax2.plot(x2,y2,color='red')
    ax2.set_title('SSIM CDF after sorting')
    ax2.set_xlabel('Difference in accuracy')
    ax2.set_ylabel('CDF of accuracy differences')

    #plt.title() 
    fig.suptitle(title, fontsize=16)
    # Adjust layout so that titles and labels don’t overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    return fig
    

 