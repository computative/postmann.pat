from numpy import loadtxt, var, mean
from matplotlib.pyplot import plot, show
from time import time

print "Loading data..."
x = loadtxt("/home/marius/Dokumenter/fys4411/postmann.pat/resources/data.txt")


print "Blocking..."
start = time()


n = len(x)
sigma = []
while(n>2):
    x = [ 0.5*(x[i] + x[i+1]) for i in range(0,n,2) ];
    n = len(x)
    sigma.append(var(x)/(n-1))
    

end = time()
print "Elapsed blocking time:", end-start
plot(sigma)
show()





"""
// maa flyttes opp
from numpy.random import choice
sigmastar = []
print "Bootstrap"
for i in range(40000):
    xstar = choice(x,100000, replace = True)
    sigmastar.append( mean(xstar) )
print var(sigmastar)
"""
