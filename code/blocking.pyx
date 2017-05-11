cdef int i, n
#cdef double xp[]
from numpy import loadtxt, var, mean
from matplotlib.pyplot import plot, show
from time import time

print "Loading data..."
x = loadtxt("/home/marius/Dokumenter/fys4411/postmann.pat/resources/data.txt")

print "Blocking..."
start = time()

n = len(x)
sigma = []
while(n > 2):
    xp = []
    for i in range(0,n,2):
        xp.append( 0.5*(x[i] + x[i+1]) )
    n = n/2
    sigma.append(var(xp)/(n-1))
    x = xp
    
end = time()
print "Elapsed blocking time:", end-start
plot(sigma)
show()
