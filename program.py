
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import math
from math import pi
from bokeh.plotting import figure, output_file, show 
import time
from numba import jit
from copy import deepcopy
import pandas as pd

start_time = time.time()

def nder(n, din, a, k):
    return din**k/(a+din**k) - n

def dder(n, d, b, h, v):
    return v*(1/(1+b*n**h) - d)

def nderRK4(n, din, a, k, step):
    k1 = step*nder(n, din, a, k)
    k2 = step*nder(n+0.5*step, din+0.5*k1, a, k)
    k3 = step*nder(n+0.5*step, din+0.5*k2, a, k)
    k4 = step*nder(n+step, din+k3, a, k)
    return n + k1/6 + k2/3 + k3/3 + k4/6

def dderRK4(n, d, b, h, v, step):
    k1 = step*dder(n, d, b, h, v,)
    k2 = step*dder(n+0.5*step, d+0.5*k1, b, h, v)
    k3 = step*dder(n+0.5*step, d+0.5*k2, b, h, v)
    k4 = step*dder(n+step, d+k3, b, h, v)
    return d + k1/6 + k2/3 + k3/3 + k4/6

def Nder1(N, Din, Rn, a, k, lam):
    return (Rn*Din**k)/(a+Din**k) - lam*N 

def Dder1(N, D, Rd, b, h, rho):
    return (Rd)/(1+b*N**h) - rho*D

def Nder2(N, D, Din, betaN, gammaN, kt, kc, MI):
    if MI == False:
        return betaN - Din*N/kt - gammaN*N
    else:
        return betaN - Din*N/kt - gammaN*N - D*N/kc
    
def Dder2(N, Nin, D, R, betaD, gammaD, kt, kc, m, MI):
    if MI == False:
        return betaD/(1+R**m) - D*Nin/kt - gammaD*D
    else:
        return betaD/(1+R**m) - D*Nin/kt - gammaD*D - D*N/kc
    
def Rder(N, Dout, R, betaR, gammaR, kRS, s):
    return betaR*(N*Dout)**s/(kRS + (N*Dout)**s) - gammaR*R

def Nder1RK4(N, Din, Rn, a, k, lam, step):
    k1 = step*Nder1(N, Din, Rn, a, k, lam)
    k2 = step*Nder1(N+0.5*step, Din+0.5*k1, Rn, a, k, lam)
    k3 = step*Nder1(N+0.5*step, Din+0.5*k2, Rn, a, k, lam)
    k4 = step*Nder1(N+step, Din+k3, Rn, a, k, lam)
    return N + k1/6 + k2/3 + k3/3 + k4/6

def Dder1RK4(N, D, Rd, b, h, rho, step):
    k1 = step*Dder1(N, D, Rd, b, h, rho)
    k2 = step*Dder1(N+0.5*step, D+0.5*k1, Rd, b, h, rho)
    k3 = step*Dder1(N+0.5*step, D+0.5*k2, Rd, b, h, rho)
    k4 = step*Dder1(N+step, D+k3, Rd, b, h, rho)
    return D + k1/6 + k2/3 + k3/3 + k4/6

def NderEuler(N, D, Din, betaN, gammaN, kt, kc, MI, step):
    return N + step*Nder2(N, D, Din, betaN, gammaN, kt, kc, MI)

def DderEuler(N, Nin, D, R, betaD, gammaD, kt, kc, m, MI, step):
    return D + step*Dder2(N, Nin, D, R, betaD, gammaD, kt, kc, m, MI)

def RderEuler(N, Dout, R, betaR, gammaR, kRS, s, step):
    return R + step*Rder(N, Dout, R, betaR, gammaR, kRS, s)

def dmean(i, line):
    sum = line[i-1]+line[i+1]
    return sum/2

def Dmean1(i, j, grid):
    sum1 = grid[i+1, j+1]
    sum1 += grid[i, j+1]
    sum1 += grid[i-1, j]
    sum1 += grid[i-1, j-1]
    sum1 += grid[i, j-1]
    sum1 += grid[i+1,j]
    return sum1/6

def Dmean2(i, j, grid, alpha):
    sum1 = grid[i+1, cvt2to1(i+1, j+1)]
    sum1 += grid[i, cvt2to1(i, j+1)]
    sum1 += grid[i-1, cvt2to1(i-1, j)]
    sum1 += grid[i-1, cvt2to1(i-1, j-1)]
    sum1 += grid[i, cvt2to1(i, j-1)]
    sum1 += grid[i+1, cvt2to1(i+1, j)]
    return sum1*alpha

def NDa(i, j, grid, wa):
    sum1 = grid[i+1, cvt2to1(i+1, j+1)]
    sum1 += grid[i, cvt2to1(i, j+1)]
    sum1 += grid[i-1, cvt2to1(i-1, j)]
    sum1 += grid[i-1, cvt2to1(i-1, j-1)]
    sum1 += grid[i, cvt2to1(i, j-1)]
    sum1 += grid[i+1, cvt2to1(i+1, j)]
    return sum1*wa

def cvt1to2(i, j):
    return j + math.floor(i/2)

def cvt2to1(i, j):
    return j - math.floor(i/2)

def cvti(i):
    return math.ceil((i+1)/2)

def StartLine(cells):
    nline = np.zeros(cells+2)
    dline = np.zeros(cells+2)
    for i in range(1, cells+1):
        nline[i] = random.normal(0.95, 0.02)
        dline[i] = random.normal(0.95, 0.02)
    return [nline, dline]

def StartGrid1(m, n):
    ngrid = np.zeros((m+2, n+2))
    dgrid = np.zeros((m+2, n+2))
    for i in range(1, m+1):
        for j in range(1, n+1):
            ngrid[i][j] = random.normal(0.95, 0.02)
            dgrid[i][j] = random.normal(0.95, 0.02)
            if ngrid[i][j] < 0:
                ngrid[i][j] = 0
            if dgrid[i][j] < 0:
                dgrid[i][j] = 0
    return [ngrid, dgrid]

def StartGrid2(m, n, pmean, psd):
    ngrid = np.zeros((m+2, n+2))
    dgrid = np.zeros((m+2, n+2))
    plengthgrid = np.zeros((m+2, n+2))
    panglegrid = np.zeros((m+2, n+2))
    pvelgrid = np.ones((m+2, n+2))
    for i in range(1, m+1):
        for j in range(1, n+1):
            ngrid[i][j] = random.normal(1, 0.01)
            dgrid[i][j] = random.normal(1, 0.01)
            plengthgrid[i][j] = random.normal(pmean, psd)
            if plengthgrid[i][j] < 0:
                plengthgrid[i][j] = 0
            panglegrid[i][j] = random.randint(0,359)*pi/180 + random.normal(0.001, 0.00001)
            if random.randint(2) == 1:
                pvelgrid[i][j] *= -1
    return [ngrid, dgrid, plengthgrid, panglegrid, pvelgrid]

def StartGrid3(m, n, Nmean, Nsd, Dmean, Dsd, plength, psd, pang, CENTRE, arcsize):
    Ngrid = np.zeros((m+2, n+2))
    Dgrid = np.zeros((m+2, n+2))
    Rgrid = np.zeros((m+2, n+2))
    plengthgrid = np.zeros((m+2, n+2))
    panglegrid = np.zeros((m+2, n+2))
    pvelgrid = np.ones((m+2, n+2))
    midpoint = cvtcart((m+2)/2, cvt1to2((m+2)/2, (n+2)/2))
    x2 = midpoint[0]
    for i in range(1, m+1):
        for j in range(1, n+1):
            Ngrid[i][j] = random.normal(Nmean, Nsd)
            Dgrid[i][j] = random.normal(Dmean, Dsd)
            plengthgrid[i][j] = random.normal(plength, psd)
            if Ngrid[i][j] < 0:
                Ngrid[i][j] = 0
            if Dgrid[i][j] < 0:
                Dgrid[i][j] = 0
            if plengthgrid[i][j] < 0:
                plengthgrid[i][j] = 0
            if pang < 0:
                panglegrid[i][j] = random.randint(0,359)*pi/180 + random.normal(0.00001, 0.00001)
            elif CENTRE == True:
                v = cvtcart(i, cvt1to2(i,j))
                x1 = v[0]
                r = dist(midpoint, v)
                if x2 == x1:
                    panglegrid[i][j] = random.randint(0,359)*pi/180 + random.normal(0.00001, 0.00001) - arcsize/2
                else:
                    perp = np.arccos((x2-x1)/r) + random.normal(0.00001, 0.00001) + pi/2
                    if perp > 2*pi:
                        perp -= 2*pi
                    if perp < 0:
                        perp += 2*pi
                    panglegrid[i][j] = perp
            else:
                panglegrid[i][j] = pang + random.normal(0.001, 0.00001)
            if random.randint(2) == 1:
                pvelgrid[i][j] *= -1
    return [Ngrid, Dgrid, Rgrid, plengthgrid, panglegrid, pvelgrid]

def StartGrid4(m, n, betaD, betaN, plength, psd):
    Ngrid = np.zeros((m+2, n+2))
    Dgrid = np.zeros((m+2, n+2))
    Rgrid = np.zeros((m+2, n+2))
    plengthgrid = np.zeros((m+2, n+2))
    panglegrid = np.zeros((m+2, n+2))
    pvelgrid = np.ones((m+2, n+2))
    for i in range(1, m+1):
        for j in range(1, n+1):
            Ngrid[i][j] = random.uniform(betaN*0.9, betaN*1.1)
            Dgrid[i][j] = random.uniform(betaD*10**-5*0.9, betaD*10**-5*1.1)
            plengthgrid[i][j] = random.normal(plength, psd)
            if Ngrid[i][j] < 0:
                Ngrid[i][j] = 0
            if Dgrid[i][j] < 0:
                Dgrid[i][j] = 0
            if plengthgrid[i][j] < 0:
                plengthgrid[i][j] = 0
            panglegrid[i][j] = random.randint(0,359)*pi/180 + random.normal(0.001, 0.00001)
            if random.randint(2) == 1:
                pvelgrid[i][j] *= -1
    return [Ngrid, Dgrid, Rgrid, plengthgrid, panglegrid, pvelgrid]
    
def sim1(cells, no, a, b, k, h, v, step):
    line = StartLine(cells)
    #line = [[0,1,0.999,0],[0,1,0.999,0]]
    t = 0; tt = [0]; fig = 0
    #nn1 = [line[0][1]]; nn2 = [line[0][2]]; dd1 = [line[1][1]]; dd2 = [line[1][2]]
    for x in range(no):
        noldline = line[0].copy()
        doldline = line[1].copy()
        
        if (x%500==0):
            plt.figure(fig)
            plt.bar(range(70), list(line[0][1:71]));
            plt.title("t = "+str(5*fig), fontsize=15)
            plt.xlabel("Cell", fontsize=15); plt.ylabel("Notch levels", fontsize=15)
            plt.xlim((-1,70)); plt.ylim((0,1)); fig += 1
        
        for i in range(1, cells+1):
            n = noldline[i]; d = doldline[i]
            line[0][i] = nderRK4(n, dmean(i, doldline), a, k, step)
            line[1][i] = dderRK4(n, d, b, h, v, step)
            if line[0][i] < 0:
                line[0][i] = 0
            if line[1][i] < 0:
                line[1][i] = 0
        t += step
        tt.append(t)
        #nn1.append(line[0][1]); nn2.append(line[0][2])
        #dd1.append(line[1][1]); dd2.append(line[1][2])
    print("Final Notch levels are:")
    print(line[0][1:cells+1])
    print("Final Delta levels are:")
    print(line[1][1:cells+1]) 
    """plt.figure(1)
    plt.plot(tt, nn1, "r", label="n1"); plt.plot(tt, nn2, "b--", label="n2");
    plt.xlabel("simulated time (arb. units)", fontsize=15); plt.ylabel("Notch levels", fontsize=15); plt.legend()
    plt.figure(2)
    plt.plot(tt, dd1, "r", label="d1"); plt.plot(tt, dd2, "b--", label="d2");
    plt.xlabel("simulated time (arb. units)", fontsize=15); plt.ylabel("Delta levels", fontsize=15); plt.legend()"""
    plt.figure(fig+1)
    plt.bar(range(70), list(line[0][1:71]));
    plt.title("t = "+str(5*fig), fontsize=15)
    plt.xlabel("Cell", fontsize=15); plt.ylabel("Notch levels", fontsize=15)
    plt.xlim((-1,70)); plt.ylim((0,1))
    
#sim1(70, 3000, 0.01, 100, 2, 2, 1, 0.01)

def plotpattern1(rows, columns, grid):
    xx = []; yy = []; colour = []
    for i in range(1, rows+1):
        for j in range(1, columns+1):
            xx.append(j); yy.append(-i)
            if grid[0][i][j] < 0.2:
                colour.append("black")
            elif grid[0][i][j] < 0.4:
                colour.append("darkgrey")
            elif grid[0][i][j] < 0.6:
                colour.append("lightgrey")
            elif grid[0][i][j] < 0.8:
                colour.append("whitesmoke")
            else:
                colour.append("white")
    graph = figure(title = "Woo", match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour)
    show(graph)
    
def plotpattern12(rows, columns, grid):
    xx = []; yy = []; colour = []
    for i in range(1, rows+1):
        for j in range(1, columns+1):
            xx.append(j); yy.append(-i)
            if grid[0][i][j] < 0.2:
                colour.append("black")
            else:
                colour.append("white")
    graph = figure(title = "Woo", match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour)
    show(graph)
    
def plotpattern2(rows, columns, grid):
    xx = []; yy = []; colour = []; Nmax = grid[0].max()
    for i in range(1, rows+1):
        for j in range(cvti(i), cvti(i)+columns):
            xx.append(j); yy.append(-i)
            if grid[0][i][cvt2to1(i, j)] < 0.2*Nmax:
                colour.append("black")
            elif grid[0][i][cvt2to1(i, j)] < 0.4*Nmax:
                colour.append("darkgrey")
            elif grid[0][i][cvt2to1(i, j)] < 0.6*Nmax:
                colour.append("lightgrey")
            elif grid[0][i][cvt2to1(i, j)] < 0.8*Nmax:
                colour.append("whitesmoke")
            else:
                colour.append("white")
    graph = figure(title = "Woo", match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour)
    show(graph)
    
def plotpattern3(rows, columns, grid):
    xx = []; yy = []; colour = []; Nmax = grid[0].max()
    for i in range(1, rows+1):
        for j in range(cvti(i), cvti(i)+columns):
            xx.append(j); yy.append(-i)
            if grid[0][i][cvt2to1(i, j)] > 0.5*Nmax:
                colour.append("white")
            else:
                colour.append("black")
    graph = figure(title = "Woo", match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour)
    show(graph)
    
def Nplotpattern1(rows, columns, grid):
    xx = []; yy = []; colour = []; Nmax = grid[0].max()
    for i in range(1, rows+1):
        for j in range(cvti(i), cvti(i)+columns):
            xx.append(j); yy.append(-i)
            if grid[0][i][cvt2to1(i, j)] < 0.2*Nmax:
                colour.append("black")
            elif grid[0][i][cvt2to1(i, j)] < 0.4*Nmax:
                colour.append("darkgrey")
            elif grid[0][i][cvt2to1(i, j)] < 0.6*Nmax:
                colour.append("lightgrey")
            elif grid[0][i][cvt2to1(i, j)] < 0.8*Nmax:
                colour.append("whitesmoke")
            else:
                colour.append("white")
    graph = figure(title = "Woo", match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour)
    show(graph)
    
def Dplotpattern1(rows, columns, grid):
    xx = []; yy = []; colour = []; Dmax = grid[1].max()
    for i in range(1, rows+1):
        for j in range(cvti(i), cvti(i)+columns):
            xx.append(j); yy.append(-i)
            if grid[1][i][cvt2to1(i, j)] > 0.8*Dmax:
                colour.append("black")
            elif grid[1][i][cvt2to1(i, j)] > 0.6*Dmax:
                colour.append("darkgrey")
            elif grid[1][i][cvt2to1(i, j)] > 0.4*Dmax:
                colour.append("lightgrey")
            elif grid[1][i][cvt2to1(i, j)] > 0.2*Dmax:
                colour.append("whitesmoke")
            else:
                colour.append("white")
    graph = figure(title = "Woo", match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour)
    show(graph)
      
def Rplotpattern1(rows, columns, grid):
    xx = []; yy = []; colour = []; Rmax = grid[2].max()
    for i in range(1, rows+1):
        for j in range(cvti(i), cvti(i)+columns):
            xx.append(j); yy.append(-i)
            if grid[2][i][cvt2to1(i, j)] < 0.2*Rmax:
                colour.append("black")
            elif grid[2][i][cvt2to1(i, j)] < 0.4*Rmax:
                colour.append("darkgrey")
            elif grid[2][i][cvt2to1(i, j)] < 0.6*Rmax:
                colour.append("lightgrey")
            elif grid[2][i][cvt2to1(i, j)] < 0.8*Rmax:
                colour.append("whitesmoke")
            else:
                colour.append("white")
    graph = figure(title = "Woo", match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour)
    show(graph)
    
def Dplotpattern2(rows, columns, grid, time):
    xx = []; yy = []; colour = []; Dmax = grid[1].max()
    for i in range(1, rows+1):
        for j in range(cvti(i), cvti(i)+columns):
            xx.append(j); yy.append(-i)
            if grid[1][i][cvt2to1(i, j)] < 0.005*Dmax:
                colour.append("white")
            elif grid[1][i][cvt2to1(i, j)] < 0.05*Dmax:
                colour.append("orange")
            elif grid[1][i][cvt2to1(i, j)] < 0.5*Dmax:
                colour.append("indianred")
            else:
                colour.append("brown")
    graph = figure(title = "t = "+str(time), match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour)
    show(graph)
    
def Dplotpattern3(rows, columns, grid, time, divwidth):
    xx = []; yy = []; colour = []; Dmax = grid[1].max(); deltano = 0
    for i in range(1, rows+1):
        for j in range(cvti(i), cvti(i)+columns):
            yy.append(-i); xx.append(j)
            if grid[1][i][cvt2to1(i, j)] < 0.005*Dmax:
                colour.append("white")
            elif grid[1][i][cvt2to1(i, j)] < 0.05*Dmax:
                colour.append("orange")
                deltano += 1
            elif grid[1][i][cvt2to1(i, j)] < 0.5*Dmax:
                colour.append("indianred")
                deltano += 1
            else:
                colour.append("brown")
                deltano += 1
    graph = figure(title = "t = "+str(time), match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour, line_width = divwidth.flatten())
    show(graph)
    return deltano
    
def Dplotpattern4(rows, columns, grid, time, divwidth):
    xx = []; yy = []; colour = []; Dmax = grid[1].max()
    for i in range(1, rows+1):
        for j in range(cvti(i), cvti(i)+columns):
            xx.append(j); yy.append(-i)
            if grid[1][i][cvt2to1(i, j)] > 0.8*Dmax:
                colour.append("black")
            elif grid[1][i][cvt2to1(i, j)] > 0.6*Dmax:
                colour.append("darkgrey")
            elif grid[1][i][cvt2to1(i, j)] > 0.4*Dmax:
                colour.append("lightgrey")
            elif grid[1][i][cvt2to1(i, j)] > 0.2*Dmax:
                colour.append("whitesmoke")
            else:
                colour.append("white")
    graph = figure(title = "t = "+str(time), match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour, line_width = divwidth.flatten())
    show(graph)
    
def Rplotpattern2(rows, columns, grid):
    xx = []; yy = []; colour = []; Rmax = grid[2].max()
    for i in range(1, rows+1):
        for j in range(cvti(i), cvti(i)+columns):
            xx.append(j); yy.append(-i)
            if grid[2][i][cvt2to1(i, j)] > 0.995*Rmax:
                colour.append("white")
            elif grid[2][i][cvt2to1(i, j)] > 0.95*Rmax:
                colour.append("orange")
            elif grid[2][i][cvt2to1(i, j)] > 0.5*Rmax:
                colour.append("indianred")
            else:
                colour.append("brown")
    graph = figure(title = "Woo", match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour)
    show(graph)
         
def sim15(rows, columns, no, a, b, k, h, v, step, PBC):
    grid = StartGrid1(rows, columns)
    nchanges = []; dchanges = []; times = []

    for x in range(no):
        
        if PBC == True:
                        
            grid[0][0][0] = grid[0][rows][columns]
            grid[1][0][0] = grid[1][rows][columns]
            
            grid[0][0][1:columns+1] = grid[0][rows][1:columns+1]
            grid[1][0][1:columns+1] = grid[1][rows][1:columns+1]
                       
            grid[0][rows+1][columns+1] = grid[0][1][1]
            grid[1][rows+1][columns+1] = grid[1][1][1]
              
            grid[0][rows+1][1:columns+1] = grid[0][1][1:columns+1]
            grid[1][rows+1][1:columns+1] = grid[1][1][1:columns+1]
                        
            for i in range(1, rows+1):
                                
                grid[0][i][0] = grid[0][i][columns]
                grid[1][i][0] = grid[1][i][columns]
               
                grid[0][i][columns+1] = grid[0][i][1]
                grid[1][i][columns+1] = grid[1][i][1]
            
        noldgrid = grid[0].copy(); doldgrid = grid[1].copy()
        nchange = 0; dchange = 0
        for i in range(1, rows+1):
            for j in range(1, columns+1):
                n = noldgrid[i][j]; d = doldgrid[i][j]
                grid[0][i][j] = nderRK4(n, Dmean1(i, j, doldgrid), a, k, step)
                grid[1][i][j] = dderRK4(n, d, b, h, v, step)
                if grid[0][i][j] < 0:
                    grid[0][i][j] = 0
                if grid[1][i][j] < 0:
                    grid[1][i][j] = 0
                nchange += abs(n-grid[0][i][j])
                dchange += abs(d-grid[1][i][j])
        nchanges.append(nchange); dchanges.append(dchange); times.append(step*x)
        if x%200 == 0:
            #plotpattern1(rows, columns, grid)
            """ngrid = list(grid[0]); dgrid = list(grid[1])
            print("Notch levels are:")
            for x in range(1, rows+1):
                print(ngrid[x][1:columns+1])
            print("Delta levels are:")
            for x in range(1, rows+1):
                print(dgrid[x][1:columns+1])"""
    ngrid = list(grid[0]); dgrid = list(grid[1])
    print("Final Notch levels are:")
    for x in range(1, rows+1):
        print(ngrid[x][1:columns+1])
    print("Final Delta levels are:")
    for x in range(1, rows+1):
        print(dgrid[x][1:columns+1])
    plotpattern1(rows, columns, grid)
    """plt.plot(times, nchanges, "b", label="Notch"); 
    plt.xlabel("simulated time (arb. units)", fontsize=15); 
    plt.ylabel("sum of absolute\n change in Notch & Delta\n concentrations", fontsize=15)
    plt.plot(times, dchanges, "r--", label="Delta")
    plt.legend()
    plt.xlim((0,no*step))"""

#sim15(50, 50, 3000, 0.01, 100, 3, 3, 1, 0.01, False)

def sim2(rows, columns, no, Rn, Rd, a, b, k, h, lam, rho, step):
    grid = StartGrid1(rows, columns)
    Nchanges = []; Dchanges = []; times = []
    for x in range(no):
        noldgrid = grid[0].copy(); doldgrid = grid[1].copy()
        Nchange = 0; Dchange = 0
        for i in range(1, rows+1):
            for j in range(1, columns+1):
                N = noldgrid[i][j]; D = doldgrid[i][j]
                grid[0][i][j] = Nder1RK4(N, Dmean1(i, j, doldgrid), Rn, a, k, lam, step)
                grid[1][i][j] = Dder1RK4(N, D, Rd, b, h, rho, step)
                if grid[0][i][j] < 0:
                    grid[0][i][j] = 0
                if grid[1][i][j] < 0:
                    grid[1][i][j] = 0
                Nchange += abs(N-grid[0][i][j])
                Dchange += abs(D-grid[1][i][j])
        Nchanges.append(Nchange); Dchanges.append(Dchange); times.append(step*x)
        #if x%200 == 0:
            #plotpattern1(rows, columns, grid)
    ngrid = list(grid[0]); dgrid = list(grid[1])
    print("Final Notch levels are:")
    for x in range(1, rows+1):
        print(ngrid[x][1:columns+1])
    print("Final Delta levels are:")
    for x in range(1, rows+1):
        print(dgrid[x][1:columns+1])
    plt.plot(times, Nchanges, "g", label="Notch"); plt.xlabel("time"); plt.ylabel("Sum of absolute change in Notch/ Delta concentration")
    plt.plot(times, Dchanges, "b--", label="Delta")
    plt.legend()
    plt.xlim((0,no*step));
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour)
    show(graph)

#sim2(5, 5, 1200, 1, 1, 0.01, 100, 2, 2, 1, 1, 0.01)   
#sim2(4, 7, 10000, 1, 5, 0.01, 100, 2, 2, 1, 1, 0.01)

def sim3(rows, columns, no, Rn, Rd, a, b, k, h, lam, rho, alpha, step):
    grid = StartGrid1(rows, columns)
    for x in range(no):
        noldgrid = grid[0].copy(); doldgrid = grid[1].copy()
        for i in range(1, rows+1):
            for j in range(cvti(i) , cvti(i)+columns):
                N = noldgrid[i][cvt2to1(i, j)]; D = doldgrid[i][cvt2to1(i, j)]
                grid[0][i][cvt2to1(i, j)] = Nder1RK4(N, Dmean2(i, j, doldgrid, alpha), Rn, a, k, lam, step)
                grid[1][i][cvt2to1(i, j)] = Dder1RK4(N, D, Rd, b, h, rho, step)
                if grid[0][i][cvt2to1(i, j)] < 0:
                    grid[0][i][cvt2to1(i, j)] = 0
                if grid[1][i][cvt2to1(i, j)] < 0:
                    grid[1][i][cvt2to1(i, j)] = 0
    ngrid = list(grid[0]); dgrid = list(grid[1])
    print("Final Notch levels are:")
    for x in range(1, rows+1):
        print(ngrid[x][1:columns+1])
    print("Final Delta levels are:")
    for x in range(1, rows+1):
        print(dgrid[x][1:columns+1])
    plotpattern2(rows, columns, grid)

def cvtcart(i, j):
    y = i
    l = 3**0.5/3
    if (i%2==0):
        x = l*(2*j - i + 1)
    else:
        x = 2*l*(j - math.floor(i/2))
    return [x, y]

def dist(v1, v2):
    return math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)

def arcstest1(r1, r2, v1, v2):
    if r1+r2 > dist(v1, v2):
        return True
    else:
        return False
    
def arcstest2(r1, r2, ang1, ang2, arcsize, v1, v2):
    d = dist(v1, v2); ang12 = ang1+arcsize; ang22 = ang2+arcsize
    if r1 > d+r2 or r2 > d+r1:
        return True
    beta = math.acos((r1**2+d**2-r2**2)/(2*r1*d)); alpha = pi/2-beta
    if v2[1] >= v1[1]:
        gamma = math.acos((v2[0]-v1[0])/d)
    else:
        gamma = 2*pi - math.acos((v2[1]-v1[1])/d)
    gamma2 = gamma+pi
    if inarc2(gamma-beta, 2*beta, v1, ang1) or inarc2(gamma-beta, 2*beta, v1, ang12):
        if inarc2(gamma2-alpha, 2*alpha, v2, ang2) or inarc2(gamma2-alpha, 2*alpha, v2, ang22):
            return True
    else:
        return False

def arcstest3(r1, r2, ang1, ang2, arcsize, v1, v2):
    m11 = math.tan(ang1); m12 = math.tan(ang1+arcsize)
    m21 = math.tan(ang2); m22 = math.tan(ang2+arcsize)
    c11 = v1[1] - m11*v1[0]; c12 = v1[1] - m12*v1[0]
    c21 = v2[1] - m21*v2[0]; c22 = v2[1] - m22*v2[0]
        
    x1 = (c21-c11)/(m11-m21); y1 = c11 + m11*x1
    if dist(v1, [x1,y1]) < r1 and dist(v2, [x1,y1]) < r2:
        return True
    x2 = (c22-c11)/(m11-m22); y2 = c11 + m11*x2
    if dist(v1, [x2,y2]) < r1 and dist(v2, [x2,y2]) < r2:
        return True
    x3 = (c21-c12)/(m12-m21); y3 = c12 + m12*x3
    if dist(v1, [x3,y3]) < r1 and dist(v2, [x3,y3]) < r2:
        return True
    x4 = (c22-c12)/(m12-m22); y4 = c12 + m12*x4
    if dist(v1, [x4,y4]) < r1 and dist(v2, [x4,y4]) < r2:
        return True
    return False
    
def arcstest4(r1, r2, ang1, ang2, arcsize, v1, v2, arcsegsize, linesegsize):
    arcseg = int(round(arcsize/arcsegsize)); lineseg = int(round(r2/linesegsize))
    ang22 = ang2+arcsize
    p = v2
    if inarc1(ang1, arcsize, v1, p) == True:
        return True
    ang = ang2
    for x in range(arcseg+1):
        p[0] = v2[0] + r2*math.cos(ang); p[1] = v2[1] + r2*math.sin(ang)
        if inarc1(ang1, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
        ang += arcsegsize
    p = v2
    hyp = r2/lineseg
    for x in range(lineseg-1):
        p[0] += hyp*math.cos(ang2); p[1] += hyp*math.sin(ang2)
        if inarc1(ang1, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
    p = v2
    for x in range(lineseg-1):
        p[0] += hyp*math.cos(ang22); p[1] += hyp*math.sin(ang22)
        if inarc1(ang1, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
    return False

def arcstest5(r1, r2, ang1, ang2, arcsize, v1, v2, arcsegsize, linesegsize):
    arcseg = int(round(arcsize/arcsegsize)); lineseg = int(round(r2/linesegsize))
    arcsegsize2 = arcsize/arcseg;
    ang12 = ang1+arcsize; ang22 = ang2+arcsize
    if ang12 > 2*pi:
        ang12 -= 2*pi
    if ang22 > 2*pi:
        ang22 -= 2*pi
    p = v2
    if inarc1(ang1, arcsize, v1, p) == True:
        return True
    if inarc1(ang12, arcsize, v1, p) == True:
        return True
    ang = ang2
    for x in range(arcseg+1):
        p[0] = v2[0] + r2*math.cos(ang); p[1] = v2[1] + r2*math.sin(ang)
        if inarc1(ang1, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
        if inarc1(ang12, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
        ang += arcsegsize2
    ang = ang2+pi
    if ang > 2*pi:
        ang -= 2*pi
    for x in range(arcseg+1):
        p[0] = v2[0] + r2*math.cos(ang); p[1] = v2[1] + r2*math.sin(ang)
        if inarc1(ang1, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
        if inarc1(ang12, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
        ang += arcsegsize2
    p = v2
    hyp = r2/lineseg
    for x in range(lineseg-1):
        p[0] += hyp*math.cos(ang2); p[1] += hyp*math.sin(ang2)
        if inarc1(ang1, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
        if inarc1(ang12, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
    p = v2
    for x in range(lineseg-1):
        p[0] += hyp*math.cos(ang22); p[1] += hyp*math.sin(ang22)
        if inarc1(ang1, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
        if inarc1(ang12, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
    p = v2
    for x in range(lineseg-1):
        p[0] += -hyp*math.cos(ang2); p[1] += -hyp*math.sin(ang2)
        if inarc1(ang1, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
        if inarc1(ang12, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
    p = v2
    for x in range(lineseg-1):
        p[0] += -hyp*math.cos(ang22); p[1] += -hyp*math.sin(ang22)
        if inarc1(ang1, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
        if inarc1(ang12, arcsize, v1, p) == True and dist(v1, p) < r1:
            return True
    return False

def arcstest6(r1, r2, ang1, ang2, arcsize, v1, v2):

    d = dist(v1, v2)
    
    if r1 > d + r2 or r2 > d + r1:
        return False 
    
    dsqrd = d**2
    rsqrdplus = r1**2 + r2**2
    rsqrdminus = r1**2 - r2**2
    con1 = rsqrdminus/(2*dsqrd)
    con2 = 0.5*(2*rsqrdplus/dsqrd - (rsqrdminus/dsqrd)**2 - 1)**0.5
    
    x1 = v1[0]; x2 = v2[0]; y1 = v1[1]; y2 = v2[1]
    
    base = 0.5*[x1+x2, y1+y2] + con1*[x2-x1, y2-y1]

    intersect1 = base + con2*[y2-y1, x1-x2]
    
    if inarc1(ang1, arcsize, v1, intersect1) == True and inarc1(ang2, arcsize, v2, intersect1):
        return True
    
    intersect2 = base - con2*[y2-y1, x1-x2]
    
    if inarc1(ang1, arcsize, v1, intersect2) == True and inarc1(ang2, arcsize, v2, intersect2):
        return True
    
    return False

def arcstest7(r1, r2, ang1, ang2, arcsize, v1, v2):
    m1 = math.tan(ang1); m2 = math.tan(ang1+arcsize)
    c1 = v1[1] - m1*v1[0]; c2 = v1[1] - m2*v1[0]
    
    delta = r2**2 * (1 + m1**2) - (v2[1] - m1*v2[0] - c1)**2
    
    if delta >= 0:
    
        basex = v2[0] + v2[1]*m1 - c1*m1
        basey = c1 + v2[0]*m1 + v2[1]*m1**2
        intersect = [(basex + delta**0.5)/(1+m1**2), (basey + m1*delta**0.5)/(1+m1**2)]

        if dist(intersect, v1) < r1 and inarc1(ang1, arcsize, v1, intersect) and inarc1(ang2, arcsize, v2, intersect):   
            return True
    
        intersect = [(basex - delta**0.5)/(1+m1**2), (basey - m1*delta**0.5)/(1+m1**2)]

        if dist(intersect, v1) < r1 and inarc1(ang1, arcsize, v1, intersect) and inarc1(ang2, arcsize, v2, intersect):   
            return True
    
    delta = r2**2 * (1 + m2**2) - (v2[1] - m2*v2[0] - c2)**2    
    
    if delta >= 0:
    
        basex = v2[0] + v2[1]*m2 - c2*m2
        basey = c2 + v2[0]*m2 + v2[1]*m2**2
        intersect = [(basex + delta**0.5)/(1+m2**2), (basey + m2*delta**0.5)/(1+m2**2)]

        if dist(intersect, v1) < r1 and inarc1(ang1, arcsize, v1, intersect) and inarc1(ang2, arcsize, v2, intersect):   
            return True
    
        intersect = [(basex - delta**0.5)/(1+m2**2), (basey - m2*delta**0.5)/(1+m2**2)]

        if dist(intersect, v1) < r1 and inarc1(ang1, arcsize, v1, intersect) and inarc1(ang2, arcsize, v2, intersect):   
            return True
        
    return False

def inarc1(ang, arcsize, v, p):
    ang1 = ang; ang2 = ang1+arcsize
    if ang2 > 2*pi:
        ang2 -= 2*pi
    angp = math.atan2(p[1]-v[1],p[0]-v[0])
    if ang1 < ang2:
        if angp > ang1 and angp < ang2:
            return True
        else:
            return False
    else:
        if angp > ang2 and angp < ang1:
            return False
        else:
            return True   
        
def inarc2(ang, arcsize, v, angp):
    ang1 = ang; ang2 = ang1+arcsize
    if ang2 > 2*pi:
        ang2 -= 2*pi
    if ang1 < ang2:
        if angp > ang1 and angp < ang2:
            return True
        else:
            return False
    else:
        if angp > ang2 and angp < ang1:
            return False
        else:
            return True 
    
def sim4(rows, columns, no, Rn, Rd, a, b, k, h, lam, rho, alpha, pmean, psd, pspeed, pprob, arcsize, 
         eff, arcsegsize, linesegsize, step):
    grid = StartGrid2(rows, columns, pmean, psd); xx = []; yy = []; colour = []
    count1 = 0; count2 = 0; count3 = 0; count4 = 0; 
    for x in range(no):
        noldgrid = grid[0].copy(); doldgrid = grid[1].copy()
        plengtholdgrid = grid[2].copy(); pangleoldgrid = grid[3].copy()
        dingrid = np.zeros((rows+2, columns+2))
        for i in range(1, rows+1):
            for j in range(cvti(i), cvti(i)+columns):
                N = noldgrid[i][cvt2to1(i, j)]; D = doldgrid[i][cvt2to1(i, j)]
                Din = Dmean2(i, j, doldgrid, alpha)
                r1 = plengtholdgrid[i][cvt2to1(i, j)]; v1 = cvtcart(i, j)
                ang1 = pangleoldgrid[i][cvt2to1(i, j)]
                for m in range(i, rows+1):
                    for n in range(cvti(m), cvti(m)+columns):
                        if m == i:
                            if n <= j:
                                continue
                        r2 = plengtholdgrid[m][cvt2to1(m, n)]; v2 = cvtcart(m, n) 
                        ang2 = pangleoldgrid[m][cvt2to1(m, n)]
                        if arcstest1(r1, r2, v1, v2) == False:
                            count1 += 1
                            continue
                        if arcstest2(r1, r2, ang1, ang2, arcsize, v1, v2) == False:
                            count2 += 1
                            continue
                        logical = arcstest3(r1, r2, ang1, ang2, arcsize, v1, v2)
                        if logical == True:
                            pass 
                        else:
                            count3 += 1
                            if arcstest4(r1, r2, ang1, ang2, arcsize, v1, v2, arcsegsize, linesegsize) == False:
                                count4 += 1
                                continue
                        dingrid[m][cvt2to1(m, n)] += eff*D
                        Din += eff*doldgrid[m][cvt2to1(m, n)]
                Din += dingrid[i][cvt2to1(i, j)]
                grid[0][i][cvt2to1(i, j)] = Nder1RK4(N, Din, Rn, a, k, lam, step)
                grid[1][i][cvt2to1(i, j)] = Dder1RK4(N, D, Rd, b, h, rho, step)
                if grid[0][i][cvt2to1(i, j)] < 0:
                    grid[0][i][cvt2to1(i, j)] = 0
                if grid[1][i][cvt2to1(i, j)] < 0:
                    grid[1][i][cvt2to1(i, j)] = 0
                if random.random() < pprob:
                   grid[2][i][cvt2to1(i, j)] = random.normal(pmean, psd)
                   grid[4][i][cvt2to1(i, j)] *= -1
                   if grid[2][i][cvt2to1(i, j)] < 0:
                       grid[2][i][cvt2to1(i, j)] = 0
                grid[3][i][cvt2to1(i, j)] += pspeed*step*grid[4][i][cvt2to1(i, j)]
                if grid[3][i][cvt2to1(i, j)] > 2*pi:
                    grid[3][i][cvt2to1(i, j)] -= 2*pi
                if grid[3][i][cvt2to1(i, j)] < 0:
                    grid[3][i][cvt2to1(i, j)] += 2*pi
                if x == no-1:
                    xx.append(j); yy.append(-i)
                    if grid[1][i][cvt2to1(i, j)] < 0.005:
                        colour.append("white")
                    elif grid[1][i][cvt2to1(i, j)] < 0.05:
                        colour.append("orange")
                    elif grid[1][i][cvt2to1(i, j)] < 0.5:
                        colour.append("indianred")
                    else:
                        colour.append("brown")
    ngrid = list(grid[0]); dgrid = list(grid[1])
    print("Final Notch levels are:")
    for x in range(1, rows+1):
        print(ngrid[x][1:columns+1])
    print("Final Delta levels are:")
    for x in range(1, rows+1):
        print(dgrid[x][1:columns+1])
    graph = figure(title = "Woo", match_aspect = True)
    graph.hex_tile(np.array(xx), np.array(yy), fill_color = colour)
    show(graph)
    print(count1, count2, count3, count4)
    
#sim4(5, 5, 10000, 1, 1, 0.01, 100, 2, 2, 1, 1, 1/6, 1.0, 0.1, 2*pi, 2, pi/2, 0.8, pi/8, 1/2, 0.01)

def sim5(size, no, step, betas, gammas, ks, powers, Ninitial, Dinitial, protrusions, 
         segsizes, ws, qs, div, MI, PRO, DIV, PBC):
    
    rows = size[0]; columns = size[1]
    betaN = betas[0]; betaD = betas[1]; betaR = betas[2]
    gammaN = gammas[0]; gammaD = gammas[1]; gammaR = gammas[2]
    kt = ks[0]; kc = ks[1]; kRS = ks[2]
    m = powers[0]; s = powers[1]
    Nmean = Ninitial[0]; Nsd = Ninitial[1]
    Dmean = Dinitial[0]; Dsd = Dinitial[1]
    
    pmean = protrusions[0]; psd = protrusions[1]; pspeed = protrusions[2]
    pproblen = protrusions[3]; pprobdir = protrusions[4]; arcsize = protrusions[5]
    MAXARC = protrusions[6]
    arcsegsize = segsizes[0]; linesegsize = segsizes[1]
    
    wa = ws[0]; wb = ws[1]; qa = qs[0]; qb = qs[1]
    
    KR = div[0]; q = div[1]; tth = div[2]
    
    grid = StartGrid3(rows, columns, Nmean, Nsd, Dmean, Dsd, pmean, psd)
    xx = []; yy = []; Ncolours = []; Dcolours = []
    count1 = 0; count2 = 0; count3 = 0; count4 = 0
    
    Ndiv = np.zeros((rows+2, columns+2)); Ddiv = np.zeros((rows+2, columns+2)); 
    divwidth = np.ones((rows, columns))
    divcount = 0
    
    #1D Experiment
    """tt1 = [0]; NN1 = [grid[0][1][1]]; DD1 = [grid[1][1][1]]; RR1 = [grid[2][1][1]]
    NN2 = [grid[0][1][2]]; DD2 = [grid[1][1][2]]; RR2 = [grid[2][1][2]]"""
    
    #MI Experiment
    """tt = []; Devolutions = []; save = 1    
    
    for x in range((rows)*(columns)):
        Devolutions.append([])"""

    #Div Experiment
    divtimes = []; Dchanges1 = []; Dchanges2 = []; ttdiv1 = []; ttdiv2 = []
    
    for x in range(no):
        
        if PBC == True:
                            
            grid[0][0][0] = grid[0][rows][columns+1]
            grid[1][0][0] = grid[1][rows][columns+1]
            grid[2][0][0] = grid[2][rows][columns+1]
            
            grid[0][0][1:columns] = grid[0][rows][1:columns]
            grid[1][0][1:columns] = grid[1][rows][1:columns]
            grid[2][0][1:columns] = grid[2][rows][1:columns]
            
            grid[0][rows+1][0] = grid[0][1][columns]
            grid[1][rows+1][0] = grid[1][1][columns]
            grid[2][rows+1][0] = grid[2][1][columns]
            
            grid[0][rows+1][1:columns] = grid[0][1][1:columns]
            grid[1][rows+1][1:columns] = grid[1][1][1:columns]
            grid[2][rows+1][1:columns] = grid[2][1][1:columns]
            
            grid[0][rows+1][columns+1] = grid[0][1][1]
            grid[1][rows+1][columns+1] = grid[1][1][1]
            grid[2][rows+1][columns+1] = grid[2][1][1]
            
            for i in range(1, rows+1):
                
                grid[0][i][0] = grid[0][i][columns]
                grid[1][i][0] = grid[1][i][columns]
                grid[2][i][0] = grid[2][i][columns]
               
                grid[0][i][columns+1] = grid[0][i][1]
                grid[1][i][columns+1] = grid[1][i][1]
                grid[2][i][columns+1] = grid[2][i][1]
        
        Noldgrid = grid[0].copy(); Doldgrid = grid[1].copy(); Roldgrid = grid[2].copy()
        plengtholdgrid = grid[3].copy(); pangleoldgrid = grid[4].copy()
        Nbgrid = np.zeros((rows+2, columns+2)); Dbgrid = np.zeros((rows+2, columns+2))
            
        #MI Experiment
        """cellno = -1; tt.append(x*step)"""
        
        #Div Experiment
        Dchange = 0; 
        
        for i in range(1, rows+1):
            for j in range(cvti(i), cvti(i)+columns):
                                        
                #cellno += 1
                                
                N = Noldgrid[i][cvt2to1(i, j)]; D = Doldgrid[i][cvt2to1(i, j)]; R = Roldgrid[i][cvt2to1(i, j)]
                
                if DIV == True:
            
                    if divwidth[i-1][cvt2to1(i, j)-1] > 1.5:
                        
                        #Devolutions[cellno].append(Ddiv[i][cvt2to1(i, j)])
                        continue
                    
                    """else:
                        
                        Devolutions[cellno].append(D)"""
                                
                if DIV == True and x*step > tth:
                    
                    probdiv = R**q/(KR**q + R**q)
                    
                    if random.random() < probdiv:
                        
                        divcount += 1
                        Ndiv[i][cvt2to1(i, j)] = N; Ddiv[i][cvt2to1(i, j)] = D; 
                        divwidth[i-1][cvt2to1(i, j)-1] = 5
                        grid[0][i][cvt2to1(i, j)] = 0; grid[1][i][cvt2to1(i, j)] = 0
                        grid[3][i][cvt2to1(i, j)] = 0
                        
                        if divcount == rows*columns:
                            no = x+1
                            
                        #Div Experiment
                        #Dplotpattern3(rows, columns, grid, x*step, divwidth)
                        divtimes.append(x*step)
                            
                        continue
                     
                Nin = NDa(i, j, Noldgrid, wa); Din = NDa(i, j, Doldgrid, wa)
                Dout = qa*Din/wa
            
                if PRO == True:  
                    
                    r1 = plengtholdgrid[i][cvt2to1(i, j)]; v1 = cvtcart(i, j)
                    ang1 = pangleoldgrid[i][cvt2to1(i, j)]
                    
                    for m in range(i, rows+1):
                        for n in range(cvti(m), cvti(m)+columns):
                            if m == i:
                                if n <= j:
                                    continue
                                
                            r2 = plengtholdgrid[m][cvt2to1(m, n)]; v2 = cvtcart(m, n) 
                            ang2 = pangleoldgrid[m][cvt2to1(m, n)]
                            
                            if MAXARC == True:
                                
                                if arcstest1(r1, r2, v1, v2) == False:
                                    continue
                                    
                            else:
                            
                                if arcstest1(r1, r2, v1, v2) == False:
                                    count1 += 1
                                    continue
                                if arcstest2(r1, r2, ang1, ang2, arcsize, v1, v2) == False:
                                    count2 += 1
                                    continue
                                if arcstest3(r1, r2, ang1, ang2, arcsize, v1, v2) == True:
                                    pass 
                                else:
                                    count3 += 1
                                    if arcstest4(r1, r2, ang1, ang2, arcsize, v1, v2, arcsegsize, linesegsize) == False:
                                        count4 += 1
                                        continue
                                
                            Nbgrid[m][cvt2to1(m, n)] += N
                            Dbgrid[m][cvt2to1(m, n)] += D
                            Nin += wb*Noldgrid[m][cvt2to1(m, n)]
                            Din += wb*Doldgrid[m][cvt2to1(m, n)]
                            Dout += qb*Doldgrid[m][cvt2to1(m, n)]
                            
                    Nin += wb*Nbgrid[i][cvt2to1(i, j)]
                    Din += wb*Dbgrid[i][cvt2to1(i, j)]; Dout += qb*Dbgrid[i][cvt2to1(i, j)] 
                    
                    if MAXARC == False:
                    
                        if random.random() < pproblen:
                            grid[3][i][cvt2to1(i, j)] = random.normal(pmean, psd)
                            if grid[3][i][cvt2to1(i, j)] < 0:
                                grid[3][i][cvt2to1(i, j)] = 0
                         
                        if random.random() < pprobdir:
                            grid[5][i][cvt2to1(i, j)] *= -1
                                
                        grid[4][i][cvt2to1(i, j)] += pspeed*step*grid[4][i][cvt2to1(i, j)]
                        if grid[4][i][cvt2to1(i, j)] > 2*pi:
                            grid[4][i][cvt2to1(i, j)] -= 2*pi
                        if grid[4][i][cvt2to1(i, j)] < 0:
                            grid[4][i][cvt2to1(i, j)] += 2*pi      
                
                grid[0][i][cvt2to1(i, j)] = NderEuler(N, D, Din, betaN, gammaN, kt, kc, MI, step)
                grid[1][i][cvt2to1(i, j)] = DderEuler(N, Nin, D, R, betaD, gammaD, kt, kc, m, MI, step)
                grid[2][i][cvt2to1(i, j)] = RderEuler(N, Dout, R, betaR, gammaR, kRS, s, step)
                                        
                if grid[0][i][cvt2to1(i, j)] < 0:
                    grid[0][i][cvt2to1(i, j)] = 0
                if grid[1][i][cvt2to1(i, j)] < 0:
                    grid[1][i][cvt2to1(i, j)] = 0
                if grid[2][i][cvt2to1(i, j)] < 0:
                    grid[2][i][cvt2to1(i, j)] = 0
                    
                Dchange += abs(D-grid[1][i][cvt2to1(i, j)])

        #1D Experiment
        """tt1.append(x*step); 
        NN1.append(grid[0][1][1]); DD1.append(grid[1][1][1]); RR1.append(grid[2][1][1])
        NN2.append(grid[0][1][2]); DD2.append(grid[1][1][2]); RR2.append(grid[2][1][2])"""
        
        #MI Experiment
        """if x%400 == 0:
            Dplotpattern1(rows, columns, grid)"""
            
        #Div Experiment
        Dchanges1.append(Dchange)
        ttdiv1.append(x*step)
        if x*step > tth:
            Dchanges2.append(Dchange)
            ttdiv2.append(x*step)
    
    if DIV == True:
        for i in range(1, rows+1):
            for j in range(cvti(i), cvti(i)+columns):
                if divwidth[i-1][cvt2to1(i, j)-1] > 1.5:
                    grid[0][i][cvt2to1(i, j)] = Ndiv[i][cvt2to1(i, j)]
                    grid[1][i][cvt2to1(i, j)] = Ddiv[i][cvt2to1(i, j)]
                                 
    #1D Experiment
    """plt.figure(1)
    plt.plot(tt1, np.log10(NN1), "r", label="N1"); plt.plot(tt1, np.log10(NN2), "b--", label="N2");
    plt.xlabel("t"); plt.ylabel("log10(Notch levels)"); plt.legend()
    plt.figure(2)
    plt.plot(tt1, np.log10(DD1), "r", label="D1"); plt.plot(tt1, np.log10(DD2), "b--", label="D2");
    plt.xlabel("t"); plt.ylabel("log10(Delta levels)"); plt.legend()
    plt.figure(3)
    plt.plot(tt1, np.log10(RR1), "r", label="R1"); plt.plot(tt1, np.log10(RR2), "b--", label="R2");
    plt.xlabel("t"); plt.ylabel("log10(Reporter levels)"); plt.legend()
    plt.figure(4)
    plt.plot(tt1, NN2, "b"); plt.xlabel("t"); plt.ylabel("Notch levels")
    plt.xlim(5,6); plt.ylim(0,1)"""
       
    #MI/Div Experiment
    """for x in range((rows)*(columns)):
        if Devolutions[x][-1] > 1:
            plt.plot(tt, np.log10(Devolutions[x]), "r", linewidth=0.5)
            plt.yscale("log")
        else:
            plt.plot(tt, np.log10(Devolutions[x]), "b", linewidth=0.5)
            plt.yscale("log")
            
    plt.xlabel("time", fontsize=12); plt.ylabel("Delta concentrations", fontsize=12)"""
    """   
    if MI == True:
        plt.title("LIMI Model", fontsize=15)
    else:
        plt.title("LI Model", fontsize=15)
        
    arrayfinal = []
    
    for y in Devolutions:
        arrayfinal.append(y[-1])
    
    CVfinal = np.std(arrayfinal)/np.mean(arrayfinal)
    
    CV = []; thomog = 0.0
    
    for x in range(len(tt)):
        
        array = []
        
        for y in Devolutions:
            array.append(y[x])
            
        cv = np.std(array)/np.mean(array)
        CV.append(cv)
        
        if cv > 0.5*CVfinal and thomog == 0.0:
            thomog = (x+1)*save*step
            
    plt.axvline(x=thomog, color="black", linestyle="dashed")
    
    plt.figure(2)
    plt.plot(tt, CV, "black")
    plt.xlabel("time", fontsize=12); plt.ylabel("Delta CV", fontsize=12)
    plt.axvline(x=thomog, color="black", linestyle="dashed")
    
    if MI == True:
        plt.title("LIMI Model", fontsize=15)
    else:
        plt.title("LI Model", fontsize=15)
        
    plt.show()"""
    
    divno = []
    for x in range(divcount):
        divno.append(x)
                
def sim6(size, no, step, betas, gammas, ks, powers, Ninitial, Dinitial, protrusions, 
         segsizes, ws, qs, div, MI, PRO, DIV, PBC, PLOT, plot, gridtest):
    
    rows = size[0]; columns = size[1]
    betaN = betas[0]; betaD = betas[1]; betaR = betas[2]
    gammaN = gammas[0]; gammaD = gammas[1]; gammaR = gammas[2]
    kt = ks[0]; kc = ks[1]; kRS = ks[2]
    m = powers[0]; s = powers[1]
    Nmean = Ninitial[0]; Nsd = Ninitial[1]
    Dmean = Dinitial[0]; Dsd = Dinitial[1]
    
    pmean = protrusions[0]; psd = protrusions[1]; pspeed = protrusions[2]
    pproblen = protrusions[3]; pprobdir = protrusions[4]; arcsize = protrusions[5]
    MAXARC = protrusions[6]; SMALLARC = protrusions[7]; DYN = protrusions[8]
    pang = protrusions[9]; CENTRE = protrusions[10]
    arcsegsize = segsizes[0]; linesegsize = segsizes[1]
    
    wa = ws[0]; wb = ws[1]; qa = qs[0]; qb = qs[1]
    
    KR = div[0]; q = div[1]; tth = div[2]
    
    if gridtest == False:
        grid = StartGrid3(rows, columns, Nmean, Nsd, Dmean, Dsd, pmean, psd, pang, CENTRE, arcsize)
    else:
        grid = deepcopy(gridtest)
    
    count1 = 0; count2 = 0; count3 = 0; count4 = 0
    
    Ndiv = np.zeros((rows+2, columns+2)); Ddiv = np.zeros((rows+2, columns+2)); 
    Rdiv = np.zeros((rows+2, columns+2))
    divwidth = np.ones((rows, columns))
    divcount = 0
    
    interact = np.zeros((rows*columns, rows+2, columns+2))
    cellnos = np.zeros((rows+2, columns+2)); cellno = 0
    
    for i in range(1, rows+1):
        for j in range(cvti(i), cvti(i)+columns):
            cellnos[i][cvt2to1(i, j)] = int(cellno)
            cellno += 1
       
    #1D Experiment
    """tt1 = [0]; NN1 = [grid[0][1][1]]; DD1 = [grid[1][1][1]]; RR1 = [grid[2][1][1]]
    NN2 = [grid[0][1][2]]; DD2 = [grid[1][1][2]]; RR2 = [grid[2][1][2]]"""
    
    #MI Experiment
    tt = []; Devolutions = []; save = 1    
    
    for x in range((rows)*(columns)):
        Devolutions.append([])

    #Div Experiment
    divtimes = []; Dchanges1 = []; Dchanges2 = []; ttdiv1 = []; ttdiv2 = []; trigger = False
    
    for x in range(no):
        
        if PBC == True:
                            
            grid[0][0][0] = grid[0][rows][columns]
            grid[1][0][0] = grid[1][rows][columns]
            grid[2][0][0] = grid[2][rows][columns]
            
            grid[0][0][1:columns+1] = grid[0][rows][1:columns+1]
            grid[1][0][1:columns+1] = grid[1][rows][1:columns+1]
            grid[2][0][1:columns+1] = grid[2][rows][1:columns+1]
            
            grid[0][rows+1][0] = grid[0][1][columns]
            grid[1][rows+1][0] = grid[1][1][columns]
            grid[2][rows+1][0] = grid[2][1][columns]
            
            grid[0][rows+1][1:columns+1] = grid[0][1][1:columns+1]
            grid[1][rows+1][1:columns+1] = grid[1][1][1:columns+1]
            grid[2][rows+1][1:columns+1] = grid[2][1][1:columns+1]
            
            grid[0][rows+1][columns+1] = grid[0][1][1]
            grid[1][rows+1][columns+1] = grid[1][1][1]
            grid[2][rows+1][columns+1] = grid[2][1][1]
            
            for i in range(1, rows+1):
                
                grid[0][i][0] = grid[0][i][columns]
                grid[1][i][0] = grid[1][i][columns]
                grid[2][i][0] = grid[2][i][columns]
               
                grid[0][i][columns+1] = grid[0][i][1]
                grid[1][i][columns+1] = grid[1][i][1]
                grid[2][i][columns+1] = grid[2][i][1]
        
        Noldgrid = grid[0].copy(); Doldgrid = grid[1].copy(); Roldgrid = grid[2].copy()
        plengtholdgrid = grid[3].copy(); pangleoldgrid = grid[4].copy()
        Nbgrid = np.zeros((rows+2, columns+2)); Dbgrid = np.zeros((rows+2, columns+2))
        #MI Experiment
        cellno = -1; tt.append(x*step)
        
        #Div Experiment
        Dchange = 0; 
        
        for i in range(1, rows+1):
            for j in range(cvti(i), cvti(i)+columns):
                
                cellno += 1               
                        
                N = Noldgrid[i][cvt2to1(i, j)]; D = Doldgrid[i][cvt2to1(i, j)]; R = Roldgrid[i][cvt2to1(i, j)]
                
                Devolutions[cellno].append(D)
                
                if DIV == True:
            
                    if divwidth[i-1][cvt2to1(i, j)-1] > 1.5:
                        
                        #Devolutions[cellno].append(Ddiv[i][cvt2to1(i, j)])
                        continue
                    """
                    else:
                        
                        Devolutions[cellno].append(D)"""
                                
                if DIV == True and x*step > tth:
                    
                    probdiv = R**q/(KR**q + R**q)
                    
                    if random.random() < probdiv:
                        
                        divcount += 1
                        Ndiv[i][cvt2to1(i, j)] = N; Ddiv[i][cvt2to1(i, j)] = D;
                        Rdiv[i][cvt2to1(i, j)] = R
                        divwidth[i-1][cvt2to1(i, j)-1] = 5
                        grid[0][i][cvt2to1(i, j)] = 0; grid[1][i][cvt2to1(i, j)] = 0
                        grid[2][i][cvt2to1(i, j)] = 0
                        
                        if divcount == rows*columns:
                            no = x+1
                            
                        #Div Experiment
                        #Dplotpattern3(rows, columns, grid, x*step, divwidth)
                        divtimes.append(x*step)
                            
                        continue
                     
                Nin = NDa(i, j, Noldgrid, wa); Din = NDa(i, j, Doldgrid, wa)
                Dout = qa*Din/wa
            
                if PRO == True:  
                    
                    r1 = plengtholdgrid[i][cvt2to1(i, j)]; v1 = cvtcart(i, j)
                    ang1 = pangleoldgrid[i][cvt2to1(i, j)]
 
                    if DYN == True:
                                       
                        for m in range(i, rows+1):
                            for n in range(cvti(m), cvti(m)+columns):
                                if m == i:
                                    if n <= j:
                                        continue
                                    
                                r2 = plengtholdgrid[m][cvt2to1(m, n)]; v2 = cvtcart(m, n)
                                ang2 = pangleoldgrid[m][cvt2to1(m, n)]
                                
                                if MAXARC == True:
                                    
                                    if arcstest1(r1, r2, v1, v2) == False:
                                        count1 += 1
                                        continue
                                        
                                else:
                                
                                    if arcstest1(r1, r2, v1, v2) == False:
                                        count2 += 1
                                        continue
                                    
                                    if arcstest3(r1, r2, ang1, ang2, arcsize, v1, v2) == True:
                                        pass
                                    else:
                                        count3 += 1
                                        if SMALLARC == True:
                                            continue
                                        elif arcstest6(r1, r2, ang1, ang2, arcsize, v1, v2) == True:
                                            count4 += 1
                                        elif arcstest7(r1, r2, ang1, ang2, arcsize, v1, v2) == True:
                                            pass
                                        elif arcstest7(r2, r1, ang2, ang1, arcsize, v2, v1) == True:
                                            pass
                                        else:
                                            continue
                                    
                                Nbgrid[m][cvt2to1(m, n)] += N
                                Dbgrid[m][cvt2to1(m, n)] += D
                                Nin += wb*Noldgrid[m][cvt2to1(m, n)]
                                Din += wb*Doldgrid[m][cvt2to1(m, n)]
                                Dout += qb*Doldgrid[m][cvt2to1(m, n)]
                                
                        Nin += wb*Nbgrid[i][cvt2to1(i, j)]
                        Din += wb*Dbgrid[i][cvt2to1(i, j)]; Dout += qb*Dbgrid[i][cvt2to1(i, j)]
                         
                        if random.random() < pproblen:
                            grid[3][i][cvt2to1(i, j)] = random.normal(pmean, psd)
                            
                            if grid[3][i][cvt2to1(i, j)] < 0:
                                grid[3][i][cvt2to1(i, j)] = 0
                    
                        if MAXARC == False:
                    
                            if random.random() < pprobdir:
                                grid[5][i][cvt2to1(i, j)] *= -1
                            
                            grid[4][i][cvt2to1(i, j)] += pspeed*step*grid[4][i][cvt2to1(i, j)]
                            if grid[4][i][cvt2to1(i, j)] > 2*pi:
                                grid[4][i][cvt2to1(i, j)] -= 2*pi
                            if grid[4][i][cvt2to1(i, j)] < 0:
                                grid[4][i][cvt2to1(i, j)] += 2*pi
                                      
                    else:
                        
                        if x == 0:

                            for m in range(i, rows+1):
                                for n in range(cvti(m), cvti(m)+columns):
                                    if m == i:
                                        if n <= j:
                                            continue
                                        
                                    r2 = plengtholdgrid[m][cvt2to1(m, n)]; v2 = cvtcart(m, n) 
                                    ang2 = pangleoldgrid[m][cvt2to1(m, n)]
                                    
                                    if MAXARC == True:
                                        
                                        if arcstest1(r1, r2, v1, v2) == False:
                                            count1 += 1
                                            continue
                                            
                                    else:
                                    
                                        if arcstest1(r1, r2, v1, v2) == False:
                                            count2 += 1
                                            continue
                                        
                                        if arcstest3(r1, r2, ang1, ang2, arcsize, v1, v2) == True:
                                            pass
                                        else:
                                            count3 += 1
                                            if SMALLARC == True:
                                                continue
                                            elif arcstest6(r1, r2, ang1, ang2, arcsize, v1, v2) == True:
                                                count4 += 1
                                            elif arcstest7(r1, r2, ang1, ang2, arcsize, v1, v2) == True:
                                                pass
                                            elif arcstest7(r2, r1, ang2, ang1, arcsize, v2, v1) == True:
                                                pass
                                            else:
                                                continue

                                    interact[int(cellnos[i][cvt2to1(i, j)])][m][cvt2to1(m, n)] = wb       
                                    interact[int(cellnos[m][cvt2to1(m, n)])][i][cvt2to1(i, j)] = wb
                                                 
                        Nbsum = np.sum(interact[int(cellnos[i][cvt2to1(i, j)])]*Noldgrid)
                        Dbsum = np.sum(interact[int(cellnos[i][cvt2to1(i, j)])]*Doldgrid)
                        Nin += Nbsum
                        Din += Dbsum
                        Dout += qb*Dbsum/wb
                             
                grid[0][i][cvt2to1(i, j)] = NderEuler(N, D, Din, betaN, gammaN, kt, kc, MI, step) #+ np.random.normal(0, (0.01*N)*0.5)
                grid[1][i][cvt2to1(i, j)] = DderEuler(N, Nin, D, R, betaD, gammaD, kt, kc, m, MI, step) #+ np.random.normal(0, (0.01*D)*0.5)
                grid[2][i][cvt2to1(i, j)] = RderEuler(N, Dout, R, betaR, gammaR, kRS, s, step) #+ np.random.normal(0, (0.01*R)*0.5)
                   
                if grid[0][i][cvt2to1(i, j)] < 0:
                    grid[0][i][cvt2to1(i, j)] = 0
                if grid[1][i][cvt2to1(i, j)] < 0:
                    grid[1][i][cvt2to1(i, j)] = 0
                if grid[2][i][cvt2to1(i, j)] < 0:
                    grid[2][i][cvt2to1(i, j)] = 0
                    
                Dchange += abs(D-grid[1][i][cvt2to1(i, j)])
                
        """if x*step > tth-step and trigger == False:
            Dplotpattern3(rows, columns, grid, x*step, divwidth)
            trigger = True"""
            

        #1D Experiment
        """tt1.append(x*step); 
        NN1.append(grid[0][1][1]); DD1.append(grid[1][1][1]); RR1.append(grid[2][1][1])
        NN2.append(grid[0][1][2]); DD2.append(grid[1][1][2]); RR2.append(grid[2][1][2])"""
        
        if plot > 0:
            if x%int(plot) == 0:
                Dplotpattern3(rows, columns, grid, x*step, divwidth)
            
        #Div Experiment
        Dchanges1.append(Dchange)
        ttdiv1.append(x*step)
        if x*step > tth:
            Dchanges2.append(Dchange)
            ttdiv2.append(x*step)
    
    if DIV == True:
        for i in range(1, rows+1):
            for j in range(cvti(i), cvti(i)+columns):
                if divwidth[i-1][cvt2to1(i, j)-1] > 1.5:
                    grid[0][i][cvt2to1(i, j)] = Ndiv[i][cvt2to1(i, j)]
                    grid[1][i][cvt2to1(i, j)] = Ddiv[i][cvt2to1(i, j)]
                    grid[2][i][cvt2to1(i, j)] = Rdiv[i][cvt2to1(i, j)]
    
    if PLOT == True:
        deltano1 = Dplotpattern3(rows, columns, grid, x*step, divwidth)
                                                                  
    #1D Experiment
    """plt.figure(1)
    plt.plot(tt1, NN1, "r", label="N1"); plt.plot(tt1, NN2, "b--", label="N2");
    plt.yscale("log")
    plt.xlabel("simulated time (arb. units)", fontsize=15); plt.ylabel("Notch levels", fontsize=15); plt.legend()
    plt.figure(2)"""
    """plt.plot(tt1, DD1, "r", label="D1"); plt.plot(tt1, DD2, "b--", label="D2");
    plt.yscale("log")
    plt.xlabel("simulated time (arb. units)", fontsize=15); plt.ylabel("Delta levels", fontsize=15); plt.legend()"""
    """plt.figure(3)
    plt.plot(tt1, RR1, "r", label="R1"); plt.plot(tt1, RR2, "b--", label="R2");
    plt.yscale("log")
    plt.xlabel("simulated time (arb. units)", fontsize=15); plt.ylabel("Reporter levels", fontsize=15);plt.legend()"""
    """plt.figure(4)
    plt.plot(tt1, NN1, "r", label="N1"); plt.plot(tt1, NN2, "b--", label="N2");
    plt.xlabel("simulated time (arb. units)", fontsize=15); plt.ylabel("Notch levels", fontsize=15);plt.legend()
    plt.xlim((5,6)); plt.ylim((0,0.8))"""
    
    #MI/Div Experiment
    """for x in range((rows)*(columns)):
        if Devolutions[x][-1] > 1:
            plt.plot(tt, Devolutions[x], "r", linewidth=0.5)
            plt.yscale("log")
        else:
            plt.plot(tt, Devolutions[x], "b", linewidth=0.5)
            plt.yscale("log")
            
    plt.xlabel("simulated time (arb. units)", fontsize=15); plt.ylabel("Delta levels", fontsize=15)
    
    if MI == True:
        plt.title("LIMI Model", fontsize=20)
    else:
        plt.title("LI Model", fontsize=20)"""
    
    arrayfinal = []
    
    for y in Devolutions:
        arrayfinal.append(y[-1])
    
    CVfinal = np.std(arrayfinal)/np.mean(arrayfinal)
    
    CV = []; thomog = 0.0
    
    for x in range(len(tt)):
        
        array = []
        
        for y in Devolutions:
            array.append(y[x])
            
        cv = np.std(array)/np.mean(array)
        CV.append(cv)
        
        if cv > 0.5*CVfinal and thomog == 0.0:
            thomog = (x+1)*save*step
       
    """plt.axvline(x=thomog, color="black", linestyle="dashed")
    
    plt.figure(2)
    plt.plot(tt, CV, "black")
    plt.xlabel("simulated time (arb. units)", fontsize=15); plt.ylabel("Delta CV", fontsize=15)
    plt.axvline(x=thomog, color="black", linestyle="dashed")
    
    if MI == True:
        plt.title("LIMI Model", fontsize=20)
    else:
        plt.title("LI Model", fontsize=20)
        
    plt.show()"""
    
    divno = []
    for x in range(divcount):
        divno.append(x)
  
    """if wb == 1 and DIV == True:
        plt.scatter(divtimes, divno, c="orange", label="(a)")
        plt.legend()
    elif DIV == True:
        plt.scatter(divtimes, divno, c="green", label="(b)")
        plt.legend()
        
    plt.title("Progression of Cell Division", fontsize=18); 
    plt.xlabel("time", fontsize=15); plt.ylabel("Divided Cell Number", fontsize=15)"""
    
    plt.title("Changes in Delta concentration over time", fontsize=18);
    plt.xlabel("simulated time (arb. units)", fontsize=15); plt.ylabel("sum of absolute\n residuals of Delta\n concentration", fontsize=15)
    """if DIV == True and tth > 2.5:
        plt.plot(ttdiv1, Dchanges1, "b--", label="Cell Division")
    elif DIV == False:
        plt.plot(ttdiv1, Dchanges1, "r", label="No Cell Division")"""
    if DYN == True:
        plt.plot(ttdiv1, Dchanges1, "b--", label="Dynamic Protrusions")
    else:
        plt.plot(ttdiv1, Dchanges1, "r", label="Non-Dynamic Protrusions")
    plt.legend()
    plt.ylim((0, 50))
    """plt.figure(3)
    plt.plot(ttdiv2, Dchanges2)
    plt.title("Changes in Delta concentration over time", fontsize=18);
    plt.xlabel("simulated time (arb. units)", fontsize=15); plt.ylabel("Sum of absolute\n residuals of Delta\n concentration", fontsize=15)"""
    
           
    """Ngrid = list(grid[0]); Dgrid = list(grid[1]); Rgrid = list(grid[2])
    print("Steps:" + str(no))
    print("Final Notch levels are:")
    for x in range(1, rows+1):
        print(Ngrid[x][1:columns+1])
    print("Final Delta levels are:")
    for x in range(1, rows+1):
        print(Dgrid[x][1:columns+1])
    print("Final Reporter levels are:")
    for x in range(1, rows+1):
        print(Rgrid[x][1:columns+1])
    print("Arcstests:")
    print(count1, count2, count3, count4)"""
    return(deltano1, thomog)
   
bN = 100; bD = 500; bR = 300000

STILL = []; DYNAMIC = []

#for x in range(5):
    
STILL.append(sim6(size=[10,10], no=2000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/20,False,True,False,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,1], qs=[1,1], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-400, gridtest=False))
    
#for x in range(5):
    
DYNAMIC.append(sim6(size=[10,10], no=2000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/20,False,True,True,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,1], qs=[1,1], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-400, gridtest=False))

"""STILLdelta = [item[0] for item in STILL]
STILLthomog = [item[1] for item in STILL]
DYNAMICdelta = [item[0] for item in DYNAMIC]
DYNAMICthomog = [item[1] for item in DYNAMIC]

print(np.mean(STILLdelta)); print(np.std(STILLdelta))
print(np.mean(STILLthomog)); print(np.std(STILLthomog))
print(np.mean(DYNAMICdelta)); print(np.std(DYNAMICdelta))
print(np.mean(DYNAMICthomog)); print(np.std(DYNAMICthomog))"""

#a
"""sim6(size=[20,20], no=4000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/8,True,False,False,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,1], qs=[1,1], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=400, gridtest=False)"""

#b
gridtest1 = StartGrid3(10, 10, 10**-3*bN, (10**-4*bN)**0.5, 10**-3*bD, (10**-4*bD)**0.5,
                      1.3, 0.3, -1, False, pi/16)
gridtest2 = StartGrid3(3, 19, 10**-3*bN, (10**-4*bN)**0.5, 10**-3*bD, (10**-4*bD)**0.5,
                      2.02, 0.2, -1, False, pi/16)
"""
sim6(size=[10,10], no=5000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/8,True,False,True,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.1], qs=[0.01,0.025], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=False, plot=200, gridtest=False)
"""
sim6(size=[20,20], no=10000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/8,True,False,True,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,1], qs=[1,1], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=False, PLOT=True, plot=200, gridtest=False)
"""
sim6(size=[3,19], no=10000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/8,True,False,False,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.1], qs=[0.01,0.025], div=[200,5,2],
     MI=True, PRO=True, DIV=True, PBC=True, PLOT=True, plot=-200, gridtest=gridtest2)"""

#c
"""sim6(size=[3,19], no=3000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/8,True,False,False,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.01], qs=[0.01,0.01], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-1, gridtest=False)"""

#d
"""sim6(size=[20,20], no=4000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/8,True,False,False,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.01], qs=[0.01,0.005], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=400, gridtest=False)"""

#e
"""sim6(size=[20,20], no=4000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/20,False,True,False,pi/6-pi/40,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.1], qs=[0.01,0.05], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=200)
"""
#e*

"""sim6(size=[3,19], no=4000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/20,False,True,False,pi/6-pi/40,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.1], qs=[0.01,0.05], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-200, gridtest=False)"""

#f
"""sim6(size=[20,20], no=5000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/20,False,True,False,pi/2-pi/40,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.2], qs=[0.01,0.2], div=[200,5,2],
     MI=True, PRO=True, DIV=True, PBC=True, PLOT=True, plot=400, gridtest=False)"""

#f*
"""sim6(size=[3,19], no=3000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/10,False,True,False,pi-pi/20,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.2], qs=[0.01,0.2], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=False, PLOT=True, plot=400, gridtest=False)"""

#g
"""sim6(size=[20,20], no=4000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[3.46,0.2,2*pi,0.1,0.1,pi/20,False,True,False,1,True], 
     segsizes=[pi/16,0.5],
     ws=[1,0.1], qs=[0.01,0.05], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=200)
"""
#h
"""sim6(size=[20,20], no=4000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/20,False,True,False,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.2], qs=[0.01,0.2], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=400, gridtest=False)"""



"""sim6(size=[20,20], no=8000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.6,0.2,2*pi,0.1,0.1,pi/8,True,False,False,pi/6,False], 
     segsizes=[pi/16,0.5],
     ws=[0.5,0.5], qs=[0.25,0.025], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=400, gridtest=False)"""
"""
sim6(size=[20,20], no=8000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/8,True,False,False,pi/6,False], segsizes=[pi/16,0.5],
     ws=[1,0.25], qs=[0.3,0.025], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=False)
  
sim6(size=[20,20], no=8000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/8,True,False,False,pi/6,False], segsizes=[pi/16,0.5],
     ws=[1,0.25], qs=[0.25,0.025], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=False)

sim6(size=[20,20], no=8000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[3.17,0.2,2*pi,0.1,0.1,pi/40,False,True,False,pi/6,False], segsizes=[pi/16,0.5],
     ws=[0.5,0.5], qs=[0.5,0.005], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=False)

sim6(size=[20,20], no=8000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/20,False,True,False,pi/6,False], segsizes=[pi/16,0.5],
     ws=[0.5,1], qs=[0.5,0.1], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=False)

sim6(size=[20,20], no=8000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[3.17,0.2,2*pi,0.1,0.1,pi/100,False,True,False,pi/6,False], segsizes=[pi/16,0.5],
     ws=[1,2], qs=[1,0.2], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=False)"""

"""sim6(size=[20,20], no=4000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/20,False,True,False,0], segsizes=[pi/16,0.5],
     ws=[1,0.1], qs=[0.01,0.05], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True)"""
    
"""

fac = 0.01; FAC1 = []

for x in range(10):
    
    FAC1.append(sim6(size=[20,20], no=1000, step=0.005, 
         betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
         Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
         protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/16,True,False,False,-1,False], 
         segsizes=[pi/16,0.5],
         ws=[1*fac,0.2*fac], qs=[1*fac,0.2*fac], div=[200,5,2],
         MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-500, gridtest=False))

fac = 0.05; FAC2 = []
    
for x in range(10):
    
    FAC2.append(sim6(size=[20,20], no=1000, step=0.005, 
         betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
         Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
         protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/16,True,False,False,-1,False], 
         segsizes=[pi/16,0.5],
         ws=[1*fac,0.2*fac], qs=[1*fac,0.2*fac], div=[200,5,2],
         MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-500, gridtest=False))
 
fac = 0.1; FAC3 = []
    
for x in range(10):
    
    FAC3.append(sim6(size=[20,20], no=1000, step=0.005, 
         betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
         Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
         protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/16,True,False,False,-1,False], 
         segsizes=[pi/16,0.5],
         ws=[1*fac,0.2*fac], qs=[1*fac,0.2*fac], div=[200,5,2],
         MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-500, gridtest=False))
    
fac = 0.5; FAC4 = []
    
for x in range(10):
    
    FAC4.append(sim6(size=[20,20], no=1000, step=0.005, 
         betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
         Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
         protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/16,True,False,False,-1,False], 
         segsizes=[pi/16,0.5],
         ws=[1*fac,0.2*fac], qs=[1*fac,0.2*fac], div=[200,5,2],
         MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-500, gridtest=False))
    
fac = 1; FAC5 = []
    
for x in range(10):
    
    FAC5.append(sim6(size=[20,20], no=1000, step=0.005, 
         betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
         Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
         protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/16,True,False,False,-1,False], 
         segsizes=[pi/16,0.5],
         ws=[1*fac,0.2*fac], qs=[1*fac,0.2*fac], div=[200,5,2],
         MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-500, gridtest=False))
    
fac = 2; FAC6 = []
    
for x in range(10):
    
    FAC6.append(sim6(size=[20,20], no=1000, step=0.005, 
         betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
         Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
         protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/16,True,False,False,-1,False], 
         segsizes=[pi/16,0.5],
         ws=[1*fac,0.2*fac], qs=[1*fac,0.2*fac], div=[200,5,2],
         MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-500, gridtest=False))
   
fac = 5; FAC7 = []
    
for x in range(10):
    
    FAC7.append(sim6(size=[20,20], no=1000, step=0.005, 
         betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
         Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
         protrusions=[2.02,0.2,2*pi,0.1,0.1,pi/16,True,False,False,-1,False], 
         segsizes=[pi/16,0.5],
         ws=[1*fac,0.2*fac], qs=[1*fac,0.2*fac], div=[200,5,2],
         MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-500, gridtest=False))

print(FAC1)
print(FAC2)
print(FAC3)
print(FAC4)
print(FAC5)
print(FAC6)
print(FAC7)

FAC1delta = [item[0] for item in FAC1]
FAC1thomog = [item[1] for item in FAC1]
FAC2delta = [item[0] for item in FAC2]
FAC2thomog = [item[1] for item in FAC2]
FAC3delta = [item[0] for item in FAC3]
FAC3thomog = [item[1] for item in FAC3]
FAC4delta = [item[0] for item in FAC4]
FAC4thomog = [item[1] for item in FAC4]
FAC5delta = [item[0] for item in FAC5]
FAC5thomog = [item[1] for item in FAC5]
FAC6delta = [item[0] for item in FAC6]
FAC6thomog = [item[1] for item in FAC6]
FAC7delta = [item[0] for item in FAC7]
FAC7thomog = [item[1] for item in FAC7]

print(FAC1delta)
print(FAC1thomog)
print(FAC2delta)
print(FAC2thomog)
print(FAC3delta)
print(FAC3thomog)
print(FAC4delta)
print(FAC4thomog)
print(FAC5delta)
print(FAC5thomog)
print(FAC6delta)
print(FAC6thomog)
print(FAC7delta)
print(FAC7thomog)

alphas = [0.01, 0.05, 0.1, 0.5, 1, 2, 5]
deltas = [FAC1delta,FAC2delta,FAC3delta,FAC4delta,FAC5delta,FAC6delta,FAC7delta]

thomogs = [FAC1thomog,FAC2thomog,FAC3thomog,FAC4thomog,FAC5thomog,FAC6thomog,FAC7thomog]

deltasmean = []; deltassd = []; thomogsmean = []; thomogssd = []

for x in range(7):
    deltasmean.append(np.mean(deltas[x]))
    deltassd.append(2*np.std(deltas[x]))
    thomogsmean.append(np.mean(thomogs[x]))
    thomogssd.append(2*np.std(thomogs[x]))

plt.figure(1)
plt.errorbar(alphas, deltasmean, yerr = deltassd, fmt="o")
plt.xlabel("alpha", fontsize=15)
plt.ylabel("Number of Delta Cells", fontsize=15)
plt.title("Number of Delta Cells to changes\n in Sensitivity of Notch Activation", fontsize=20)

plt.figure(2)
plt.errorbar(alphas, thomogsmean, yerr = thomogssd, fmt="o")
plt.xlabel("alpha", fontsize=15)
plt.ylabel("Homogeneous Time", fontsize=15)
plt.title("Homogeneous Time to changes\n in Sensitivity of Notch Activation", fontsize=20)

"""


#DIVISION
""" 
sim6(size=[10,10], no=1200, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.3,0.2,2*pi,0.1,0.1,pi/16,False,True,True,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.2], qs=[1,0.2], div=[200,5,2],
     MI=True, PRO=True, DIV=False, PBC=True, PLOT=True, plot=-1, gridtest=gridtest1)
"""

"""sim6(size=[10,10], no=3000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.3,0.2,2*pi,0.1,0.1,pi/16,False,True,True,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.2], qs=[1,0.2], div=[200,5,2],
     MI=True, PRO=True, DIV=True, PBC=True, PLOT=True, plot=-1, gridtest=gridtest1)"""

"""sim6(size=[10,10], no=4000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[2.3,0.2,2*pi,0.1,0.1,pi/16,False,True,True,-1,False], 
     segsizes=[pi/16,0.5],
     ws=[1,0.2], qs=[1,0.2], div=[200,5,14],
     MI=True, PRO=True, DIV=True, PBC=True, PLOT=True, plot=-1, gridtest=gridtest1)"""
    
"""sim6(size=[20,20], no=2500, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[1.1,0.3,2*pi,0.1,0.1,pi/8,True,False], segsizes=[pi/16,0.5],
     ws=[1,1], qs=[1,1], div=[200,5,2],
     MI=False, PRO=False, DIV=False, PBC=False)"""


"""sim6(size=[10,20], no=1000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[1.1,0.3,2*pi,0.1,0.1,pi/8,True,False], segsizes=[pi/16,0.5],
     ws=[1,0.2], qs=[1,0.2], div=[200,5,2],
     MI=True, PRO=True, DIV=True, PBC=True)"""

"""for x in range(5):
    sim5(size=[10,20], no=800, step=0.005, 
         betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,10**7], powers=[2,2], 
         Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
         protrusions=[1.1,0.3,2*pi,0.1,0.1,pi/8,True], segsizes=[pi/16,0.5],
         ws=[1,0.2], qs=[1,0.2], div=[200,5,2],
         MI=True, PRO=True, DIV=True, PBC=False)""" 

"""sim5(size=[10,10], no=2000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[1.0,0.1,2*pi,0.1,0.1,pi/8,True], segsizes=[pi/16,0.5],
     ws=[1,1], qs=[1,1], div=[200,5],
     MI=True, PRO=False, DIV=False, PBC=True)"""

#MI Experiment

"""A = []

for x in range(1):
    A.append(sim6(size=[12,12], no=2400, step=0.005, 
         betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
         Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
         protrusions=[1.0,0.1,2*pi,0.1,0.1,pi/8,True,False,False,-1,False], 
         segsizes=[pi/16,0.5],
         ws=[1,1], qs=[1,1], div=[200,5,2],
         MI=False, PRO=False, DIV=False, PBC=True, PLOT=False, plot=-1))
    
B = []

for x in range(1):
    B.append(sim6(size=[12,12], no=2400, step=0.005, 
         betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[2,2], 
         Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
         protrusions=[1.0,0.1,2*pi,0.1,0.1,pi/8,True,False,False,-1,False], 
         segsizes=[pi/16,0.5],
         ws=[1,1], qs=[1,1], div=[200,5,2],
         MI=True, PRO=False, DIV=False, PBC=True, PLOT=False, plot=-1))

print(np.mean(A))
print(np.mean(B))"""

"""bN = 10*2; bD = 10**4; bR = 300000
sim5(size=[5,5], no=2000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[2,0.5,10**7], powers=[1,1], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[1.0,0.1,2*pi,0.1,0.1,pi/8,True], segsizes=[pi/16,0.5],
     ws=[1,1], qs=[1,1], div=[200,5],
     MI=False, PRO=False, DIV=False, PBC=True)"""

bD = 10; bR = 1000000
"""sim5(size=[5,5], no=4000, step=0.005, 
     betas=[bN,bD,bR], gammas=[1,1,1], ks=[1,1,300000], powers=[1,3], 
     Ninitial=[10**-3*bN, (10**-4*bN)**0.5], Dinitial=[10**-3*bD, (10**-4*bD)**0.5], 
     protrusions=[1.0,0.1,2*pi,0.1,0.1,pi/8,True], segsizes=[pi/16,0.5],
     ws=[1,1], qs=[1,1], div=[200,5],
     MI=False, PRO=False, DIV=False)"""

print("My program took", time.time() - start_time, "to run")