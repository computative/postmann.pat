#ifndef MATHS_H
#define MATHS_H

static int factorial(int n)
{
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

#endif // MATHS_H
