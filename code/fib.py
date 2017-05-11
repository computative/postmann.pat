import fib

def fibo(n):
    """Print the Fibonacci series up to n."""
    a, b = 0, 1
    while b < n:
        print b,
        a, b = b, a + b

fibo(2000)
fib.fib(2000)
