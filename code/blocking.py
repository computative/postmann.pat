from numpy import *
from matplotlib.pyplot import plot, show, title, xlabel, ylabel, savefig
from time import time

# files to block
files = [
    "sampledata.txt"
]

for entry in files:
    print "Loading file..."
    loadstart = time()
    data = loadtxt("/home/marius/Dokumenter/fys4411/postmann.pat/resources/"+entry)

    x = data[ : 2**floor(log2(len(data)))]
    print "Blocking..."
    blockstart = time()


    n = len(x)
    sigma = []
    while(n>2):
        x = [ 0.5*(x[i] + x[i+1]) for i in range(0,n,2) ];
        n = len(x)
        sigma.append(var(x)/(n-1))
    

    end = time()
    print "Elapsed importing:", blockstart-loadstart, " Elapsed blocking:", end-blockstart
    plot(sigma)
    title("Filename: %s" % entry)
    xlabel("Iteration")
    ylabel("Variance estimate")
    savefig("../benchmark/blocking.png")
