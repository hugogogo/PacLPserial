#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#include "mt19937p.h"

/* References: Zhu & Orecchia (2014) serial paper*/
/* Based on Dylan's code */
/* Currently getting bad results with float - seems like we might be dealing with values that are too small */

#define real double
#define real_max DBL_MAX
#define real_eps DBL_EPSILON

#define AA(i,j) a[i +j*m]
/* b is a length-m vector */
/* c is a length-n vector*
/* a is an mxn matrix in column major format */

void print_vec( real const * const x, int const n, int const stride)
{
	for(int idx = 0; idx < n-1; ++idx)
		printf("%f, ", x[idx*stride]);
	printf("%f\n", x[(n-1)*stride]);
}

void print_mat(real  const * const a, int const m, int const n)
{
	for (int ii = 0; ii < m; ++ii)
	{
		print_vec(a+ii, n, m);
	}
}

/*  Generates a random A matrix where entries are nonzero w/p p. */
/*  Except for first row, which has uniform entries */
real * random_instance(int const m, int const n, real const p)
{

	/* int *l = (int*) _mm_malloc(n*n*sizeof(int), ALIGNBY); */
	real *l = (real*) malloc(m*n*sizeof(real));
	struct mt19937p state;
	sgenrand(10302011UL, &state);
	for (int i = 0; i < m; ++i){
		for (int j = 0; j < n; ++j) 
		{
			if( i == 0)
				l[j*m+i] = genrand(&state);
			else 
				l[j*m+i] = (genrand(&state) < p);
		}
	}
	return l;
}

int count_nonzeros(real const * const x, int const n)
{
	int count = 0;

	for(int ii = 0; ii < n; ++ii)
	{
		if( fabs(x[ii]) > real_eps)
			++count;
	}
	return count;
}

/* Initializes variables to specified size */
void gen_test_problem(real ** A, real ** B, 
		real ** C, int const m, int const n)
{
	real * a;
	real * b;
	real * c;

	a = (real*) malloc(n*m*sizeof(real));
	b = (real*) malloc(m*sizeof(real));
	c = (real*) malloc(n*sizeof(real));

	for(int idx = 0; idx < n; ++idx)
	{
		for(int idy = 0; idy < m; ++idy)
		{
			AA(idy,idx) = 1;
		}
	}

	for(int idx = 0; idx < m; ++idx)
	{
		b[idx] = 1;
	}

	for(int idx = 0; idx < n; ++idx)
	{
		c[idx] = 1;
	}

	*A = a;
	*B = b;
	*C = c;
}

/* Convert to problem of the form, say max 1^{T}x s.t. Ax \leq{} 1 */
void to_standard_form(real * const a, real const *const b, real const * const c, 
		int const m, int const n)
{
	/* Scale down by b */
	for(int ii = 0; ii < m; ++ii)
	{
		for (int jj = 0; jj < n; ++jj)
		{
			AA(ii,jj) = AA(ii,jj) / b[ii];
		}
	}

	/* Scale down by c */
	for(int ii = 0; ii < m; ++ii)
	{
		for (int jj = 0; jj < n; ++jj)
		{
			AA(ii,jj) = AA(ii,jj) / c[jj];
		}
	}
}

/* Convert back to  max c^{T}x s.t. Ax \leq{} b */
void from_standard_form(real * const a, real const *const b, real const * const c,
		int const m, int const n)
{
	/* Scale up by b */
	for(int ii = 0; ii < m; ++ii)
	{
		for (int jj = 0; jj < n; ++jj)
		{
			AA(ii,jj) = AA(ii,jj) * b[ii];
		}
	}

	/* Scale up by c */
	for(int ii = 0; ii < m; ++ii)
	{
		for (int jj = 0; jj < n; ++jj)
		{
			AA(ii,jj) = AA(ii,jj) * c[jj];
		}
	}
}

real infinity_norm(real const * const x, int const n, int const stride)
{
	assert (stride > 0);

	real lb = 0;

	for(int idx = 0; idx < n; ++idx)
	{
		real const v = fabs(x[idx*stride]);
		if(v>lb)
			lb = v;
	}
	return lb;
}

/* Compute Ax given A and x, where A is mxn and x is length-n*/
void matvec_multr(real const * const x, real const * const a, real * const ax, int const m, int const n)
{
	for(int ii = 0; ii < m; ++ii)
	{
		ax[ii] = 0.0;
	}

	for(int jj = 0; jj < n; ++jj)
	{
		for(int ii = 0; ii < m; ++ii)
		{
			ax[ii] += AA(ii,jj)*x[jj];
		}
	}
}

/* Compute x^{T}A given A and x, where A is mxn and x is length-m */
void matvec_multl(real const * const x, real const * const a, real * const xa, int const m, int const n)
{
	for(int ii = 0; ii < m; ++ii)
	{
		xa[ii] = 0.0;
	}

	for(int jj = 0; jj < n; ++jj)
	{
		for(int ii = 0; ii < m; ++ii)
		{
			xa[jj] += AA(ii,jj)*x[ii];
		}
	}
}

/* Exponentiate vector inplace (i.e., x <- e^(scale*(x+offset))*/
void exp_vec(real * const x, real const scale, real const offset, int const n)
{
	for (int ii = 0; ii < n; ++ii)
	{
		x[ii] = exp(scale*(x[ii]+offset));
	}
}

void LC_vec(real * res, real * const x1, real * const x2, real const s1, real const s2, int const n)
{
	/* compute linear combination of x1 and x2 given s1 and s2
	 * i.e. res = s1 * x1 + s2 * s2 */
	for (int ii = 0; ii < n; ++ii)
		res[ii] = s1*x1[ii]+s2*x2[ii];
}

void copy_vec(real * const x1, real * const x2, int const n){
	/* copy vector x2 to x1, i.e., x1 <- x2 */
	for (int ii = 0; ii < n; ++ii)
		x1[ii] = x2[ii];	
}

real crossprod(real * const x, real const * const y, int const n){
	real res = 0;
	for (int ii = 0; ii < n; ++ii){
		res += x[ii]*y[ii];	
	}
	return res;
}

/* x1 <- scale * x1 */
void scale_vec(real * const x, real const scale, int const n)
{
	for (int ii = 0; ii < n; ++ii)
	{
		x[ii] *=scale;
	}
}

/* a is assumed to be column-major */
void scale_mat(real * const a, real const scale, int const m, int const n)
{
	for (int ii = 0; ii < n; ++ii)
	{
		scale_vec(a+ii*m, scale, m);
	}
}

real thresh(real const * const a, real * const x, real * const tmp, real const mu, int ind, int const m, int const n)
{
  	real v;
	/* Threshold operator in pp.5 and 6 */
	/* tmp = Ax */
	matvec_multr(x, a, tmp, m, n);
	/* tmp = exp((tmp - 1)/mu)*/
	exp_vec(tmp, 1/mu, -1, m);
	/* v = tmp ^t A[,ind] - 1 */
	v = crossprod(tmp, a+ind*m, m) - 1;
	return v > 1? 1 : v;
}

real mirror(real inf, real zi, real delta){
	/* performs a mirror update */
	real res = zi - delta/inf;
	if (res < 0)
		res = 0;
	if (res > 1/inf)
		res = 1/inf;
	return res;

}
/* Solve standard form LP */
/* Output should be (1-O(eps))-approximately optimal and
 * satisfies constraints precisely */
void PacLPsolver(real ** Y, real const epsi, real const * const a, int const m, int const n)
{
	/* Arg:
	 * a: an m-by-n matrix with non-negative entries 
	 */

	/* Alg. in paper doesn't solve to (1-eps) but rather (1-O(eps)). This guarantees our solution is (1-eps)-optimal */
	real const eps = epsi/4;	

	/* Set seed for random number generator*/
	srand((unsigned) time(NULL));

	/* Solution variables */
	real * x;
	real * y; 
	real * z;
	real * tmp;
	real * infnorm;

	x = (real*) malloc(n*sizeof(real)); 
	y = (real*) malloc(n*sizeof(real)); 
	z = (real*) malloc(n*sizeof(real)); 
	tmp = (real*) malloc(n*sizeof(real));
	infnorm = (real*) malloc(n*sizeof(real));

	memset(z, 0, n*sizeof(real));

	/* Constants */
	real const mu = eps/(4*log(n*m/eps));
	real const L = 4/mu;
	real const tau = 1/(3*n*L);
	/* changing alpha in each iteration */
	real alpha = 1/(n*L);
	long int const T = ((int) 3*n*L*log(1/eps)+1);
	/* uniform random index */
	int ind;
	/* xi: tmp real variable */
	real xi;
	real tmpreal;

	printf("Running for %d iterations.\n", T);

	/* Initial value of x and y in \Delta, see Fact 2.7 pp.6 */
	/* See pp.4 for the definition of \Delta */
	/* Also stores the inf norm of columns of A for later use*/
	for(int ii = 0; ii < n; ++ii)
	{
		real const ainf = infinity_norm(a+ii*m, m, 1);
		infnorm[ii] = ainf;
		x[ii] = (1-eps/2)/(n*ainf);
		y[ii] = (1-eps/2)/(n*ainf); 
	}

	double start = omp_get_wtime();

	/* Main loop */
	for(long int t = 0; t < T; ++t)
	{
		if( !(t % 5000000))
		{
			double now = omp_get_wtime();

			printf("Reached iteraton %d (%f%%). Elapsed time %f\n", t, (float)t/T*100, now-start);
			printf("Primal: ");
			print_vec(x, n, 1);
			fflush(stdout);
		}

		/* update alpha_k */
		alpha = alpha/(1 - tau);

		/* update x_k <- tau z_k + (1-tau) y_k */
		LC_vec(x, z, y, tau, (1-tau), n);
		/* y <- x */
		copy_vec(y, x, n);

		/* choose index from uniform (1,n) */
		ind = rand() % n; 

		/* Compute Threshold function to update xi_k^i */
		xi = thresh(a, x, tmp, mu, ind, m, n);

		/* Mirror Step */
		tmpreal = mirror(infnorm[ind], z[ind], n*alpha*xi);

		/* update x, y, z*/
		y[ind] = x[ind] + 1/(n*alpha*L)*(tmpreal - z[ind]);
		z[ind] = tmpreal;
	}

	/* At this point, y should be (1+4eps)OPT, so at most (1+epsi)OPT (since we scaled eps down) */

	free(tmp);
	free(z);
	free(infnorm);
	free(x);

	*Y = y;
}

real min_colinf(real const *const a, int const m, int const n)
{
	real ub = real_max;
	for(int ii = 0; ii < n; ++ii)
	{
		real const ainf = infinity_norm(a+ii*m, m, 1);
		if (ub >= ainf)
			ub = ainf;
	}
	return ub;
}

void certify_standard_form(real const* const a, real const * const x, real const eps,
		int const m, int const n)
{
	real p = 0;

	real * ax = (real*) malloc(m*sizeof(real));
	matvec_multr(x, a, ax, m,n);

	for(int ii = 0; ii < m; ++ii)
	{
		if (ax[ii] > 1 + real_eps)
		{
			printf("Primal solution violates constraint %d (Value %f)\n", ii, ax[ii]);
		}
	}
	for(int ii =0; ii < n; ++ii)
	{
		p += x[ii];
	}

	printf("Primal value: %f", p);

	free(ax);
}

/* Solve packing LPs of the form max c^{T}x subject to Ax <= b */
int main(int argc, char **argv)
{
	int n = 5;			/* Dimensionality */
	int m = 5;			/* Number of constraints */
	int N = 0;			/* Number of non-zeros */
	real eps = 0.099;		/* precision */
	real *a;		/* instance */
	real *y; 			/* solution variables */


	/* Read in parameters */

	printf("Variables: %d, Constraints: %d\n", n, m);

	a = random_instance(m,n, 0.6);
	N = count_nonzeros(a,m*n);
	printf("%d non-zeros in A.\n", N);

	/* Get test instance and convert to standard LP formulation */
	/* gen_test_problem(&a,&b,&c,m,n); */
	/* Put in standard packing LP form */
	/* to_standard_form(a,b,c,m,n); */
	/* This will totally break if b and c have zeros. Have to do a different reduction in this case */

	printf("A:\n");
	print_mat(a,m,n);

	/* Scale A down by infinity norm (as per page 5)  */
	/* Assumes A has no all-zero columns */
	/* real ainf = min_colinf(a, m,n);
	scale_mat(a, 1/ainf, m, n); */

	/* Solve LP */
	assert(eps > 0 && eps <= 0.1);

	printf("Solving to precision %f.\n", eps);

	PacLPsolver(&y,eps,a,m,n);
	/* Expect x and y to satisfy constraints perfectly and to have */
	/* 1^{T}x >= (1-eps)OPT, 1^{T}y <= (1+eps)OPT */
	printf("Primal solution:\n");
	print_vec(y,n,1);

	/* Maybe certify x using y here */
	certify_standard_form(a, y, eps, m, n);

	/* Scale up solution (undo standardization and ainf scaling) */
	/*for(int ii = 0; ii < n; ++ii)
	{
		y[ii] *= ainf;
	}*/

	free(y);
	free(a);
	return 0;
}
