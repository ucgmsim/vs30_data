import functools
import math


import numpy as np
import pandas as pd
from qcore import geo


def nztm_to_ll(nztm_x,nztm_y):
    #Common variables for NZTM2000
    a = 6378137;
    f = 1 / 298.257222101;
    phizero = 0;
    lambdazero = 173;
    Nzero = 10000000;
    Ezero = 1600000;
    kzero = 0.9996;

    #input Northing(Y); Easting(X) variables
    #  N  = int(nztm_n);
    #  E  = int(nztm_e);
    N = nztm_y
    E = nztm_x

    #Calculation: From NZTM to lat/Long
    b = a * (1 - f);
    esq = 2 * f - f ** 2;
    Z0 = 1 - esq / 4 - 3 * (esq ** 2) / 64 - 5 * (esq ** 3) / 256;
    A2 = 0.375 * (esq + esq ** 2 / 4 + 15 * (esq ** 3) / 128);
    A4 = 15 * ((esq ** 2) + 3 * (esq ** 3) / 4) / 256;
    A6 = 35 * (esq ** 3) / 3072;

    Nprime = N - Nzero;
    mprime = Nprime / kzero;
    smn = (a - b) / (a + b);
    G = a * (1 - smn) * (1 - (smn ** 2)) * (1 + 9 * (smn ** 2) / 4 + 225 * (smn ** 4) / 64) * math.pi/ 180.0;
    sigma = mprime * math.pi / (180 * G);
    phiprime = sigma + (3 * smn / 2 - 27 * (smn ** 3) / 32) * math.sin(2 * sigma) + (21 * (smn ** 2) / 16 - 55 * (smn ** 4) / 32) * math.sin(4 * sigma) + (151 * (smn ** 3) / 96) * math.sin(6 * sigma) + (1097 * (smn ** 4) / 512) *math.sin(8 * sigma);
    rhoprime = a * (1 - esq) / ((1 - esq * ((math.sin(phiprime)) ** 2)) ** 1.5);
    upsilonprime = a / math.sqrt(1 - esq * ((math.sin(phiprime)) ** 2));

    psiprime = upsilonprime / rhoprime;
    tprime = math.tan(phiprime);
    Eprime = E - Ezero;
    chi = Eprime / (kzero * upsilonprime);
    term_1 = tprime * Eprime * chi / (kzero * rhoprime * 2);
    term_2 = term_1 * (chi ** 2) / 12 * (-4 * (psiprime ** 2) + 9 * psiprime * (1 - (tprime ** 2)) + 12 * (tprime ** 2));
    term_3 = tprime * Eprime * (chi ** 5) / (kzero * rhoprime * 720) * (8 * (psiprime ** 4) * (11 - 24 * (tprime ** 2)) - 12 * (psiprime ** 3) * (21 - 71 * (tprime ** 2)) + 15 * (psiprime ** 2) * (15 - 98 * (tprime ** 2) + 15 * (tprime ** 4)) + 180 * psiprime * (5 * (tprime ** 2) - 3 * (tprime ** 4)) + 360 * (tprime ** 4));
    term_4 = tprime * Eprime * (chi ** 7) / (kzero * rhoprime * 40320) * (1385 + 3633 * (tprime ** 2) + 4095 * (tprime ** 4) + 1575 * (tprime ** 6));
    term1 = chi * (1 / math.cos(phiprime));
    term2 = (chi ** 3) * (1 / math.cos(phiprime)) / 6 * (psiprime + 2 * (tprime ** 2));
    term3 = (chi ** 5) * (1 / math.cos(phiprime)) / 120 * (-4 * (psiprime ** 3) * (1 - 6 * (tprime ** 2)) + (psiprime ** 2) * (9 - 68 * (tprime ** 2)) + 72 * psiprime * (tprime ** 2) + 24 * (tprime ** 4));
    term4 = (chi ** 7) * (1 / math.cos(phiprime)) / 5040 * (61 + 662 * (tprime ** 2) + 1320 * (tprime ** 4) + 720 * (tprime ** 6));

    latitude = (phiprime - term_1 + term_2 - term_3 + term_4) * 180 / math.pi;
    longitude = lambdazero + 180 / math.pi * (term1 - term2 + term3 - term4);

    return (latitude,longitude)


def locs_multiple_records(cpt_locs, min_dist_m, stdout=False):

    x=[]
    y=[]
    cpt_name=[]

    for cpt in cpt_locs:
        (lat,lon) = nztm_to_ll(cpt.nztm_x, cpt.nztm_y)
        x.append(lon)
        y.append(lat)
        cpt_name.append(cpt.name)

    locs=np.array([x,y]).T

    dup_idx={}
    dup_dist={}
    dup_locs = {}

    for i in range(len(x)):
        skip_i=False
        for key in dup_idx:
            if i in dup_idx[key]:
                #i has been already marked as a duplicate
                #no need to carry on testing i
                skip_i=True
                break
        if skip_i:
            continue

        j=i+1
        while j < len(x):
            idx,d=geo.closest_location(locs[j:],x[i],y[i])
            k=j+idx
            #if d<0.0001: #0.1m may be a bit too strict
            if d < min_dist_m/1000.0: # dividing by 1000.0 to convert m to km

            #    print("{} is close to {} distance {}".format(cpt_name[i],cpt_name[k],d))
                if dup_idx.get(i) is None:
                    dup_idx[i]=[k]
                    dup_dist[i]=[d]
                    dup_locs[cpt_name[i]]=[cpt_name[k]]
                else:
                    dup_idx[i].append(k)
                    dup_dist[i].append(d)
                    dup_locs[cpt_name[i]].append(cpt_name[k])

                j=k+1
            else:
                break #no duplicate found. need to carry on searching for this

    if stdout:
        for i in dup_idx:
            print("{:12s}      {} {}".format(cpt_name[i],x[i],y[i]))
            for j in range(len(dup_idx[i])):
                k=dup_idx[i][j]
                d=dup_dist[i][j]
                print("==={}. {:12s} {} {} (dist:{:4f})".format(j+1, cpt_name[k],x[k],y[k],d))

    return dup_locs

