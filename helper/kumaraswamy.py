import torch
import numpy as np
from scipy.optimize import curve_fit
# In this distribution a controls the "stretching" (the steepness of the curve)
# b controls the "skewness" or asymmetry of the distribution
# Larger values of a and b make the distribution more skewed or concentrated towards certain regions of the support

def kumaraswamy_rvs(size,a,b):
    tensor = np.random.uniform(0,1,size)

    x = (1-(1-tensor)**(1/b))**(1/a)
    x = torch.tensor(x)
    return x

def kumaraswamy_pdf(x,a,b):
    return a*b*x**(a-1)*(1-x**a)**(b-1)

def curve_fitting(image, bins = 30, density= True):
    hist, bin_edges = np.histogram(image,bins=30,density=True)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2

    params, _ = curve_fit(kumaraswamy_pdf,bin_centers,hist)
    return params