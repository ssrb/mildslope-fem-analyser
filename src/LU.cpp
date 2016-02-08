// Copyright (c) 2015, Sebastien Sydney Robert Bigot
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the FreeBSD Project.
#include "LU.h"
#include <cstdlib>

// Pardiso + vectorized math
extern "C" {

	void pardisoinit(void*, int*, int*, int*, double*, int*);

	void pardiso(void*, int*, int*, int*, int*, int*, double*, int*, int*, int*, int*, int*, int*, double*, double*, int*, double*);

	void pardiso_chkmatrix(int*, int*, double*, int*, int*, int*);

	void pardiso_chkvec(int*, int*, double*, int*);

	void pardiso_printstats (int*, int*, double*, int*, int*, int*, double*, int*);
}

// /* Check license of the solver and initialize the solver */
// pardisoinit(pt, &mtype, &solver, iparm, dparm, &error);
//  Solve matrix sytem 
// pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja,
// perm, &nrhs, iparm, &msglvl, b, x, &error, dparm)

// typedef struct {double re; double im;} doublecomplex;
// void pardiso_chkmatrix (int *mtype, int *n, double *a,
// int *ia, int *ja, int *error);
// void pardiso_chkmatrix_z (int *mtype, int *n, doublecomplex *a,
// int *ia, int *ja, int *error);
// void pardiso_chkvec (int *n, int *nrhs, double *b, int *error);
// void pardiso_chkvec_z (int *n, int *nrhs, doublecomplex *b, int *error);
// void pardiso_printstats (int *mtype, int *n, double *a, int *ia,
// int *ja, int *nrhs, double *b, int *error);
// void pardiso_printstats_z (int *mtype, int *n, doublecomplex *a, int *ia,
// int *ja, int *nrhs, doublecomplex *b, int *error);

namespace BabyHares {
	LUConstPtr LU::Factorize(const SparseMatrix<std::complex<double> > &mat) {

		int error = 0;

		LUPtr lu(new LU);

		lu->_mtype = 6; // complex and symmetric

		int solver = 0; // use sparse direct solver

		pardisoinit(lu->_pardisoHandle, &lu->_mtype, &solver, lu->_iparm, lu->_dparm, &error);

		lu->_iparm[2] = 1; // # processors : use OMP_NUM_THREADS
		const char *omp_num_threads = getenv("OMP_NUM_THREADS");
		if (omp_num_threads) {
			lu->_iparm[2] = atoi(omp_num_threads);
		}
		//pardisoLU.iparm[26] = 1; // Check input
		lu->_iparm[28] = 0; // Pardiso double precision

		// Intel doc : Maximum number of factors with identical nonzero sparsity structure 
		// that must be keep at the same time in memory. 
		// In most applications this value is equal to 1.
		int maxfct = 1;

		// Intel doc: Indicates the actual matrix for the solution phase. 
		// With this scalar you can define which matrix to factorize.
		// In most applications this value is 1.
		int mnum = 1;

		int phase = 12; // Analysis, numerical factorization

		lu->_n = mat.nrow;
		
		// Intel doc: Holds the permutation vector of size n. 
		// You can use it to apply your own fill-in reducing ordering to the solver. 
		// The permutation vector perm is used by the solver if iparm(5) = 1.
		int *perm = 0;

		// Number of right-hand sides that need to be solved for.
		int nrhs = 0;

		int msglvl = 0; // Debug output => 1

		lu->_val = &mat.val[0];
		lu->_rowPtr = &mat.rowPtr[0];
		lu->_col = &mat.col[0];

		pardiso(lu->_pardisoHandle,
				&maxfct,
				&mnum,
				&lu->_mtype,
				&phase,
				&lu->_n,
				const_cast<double *>(reinterpret_cast<const double *>(&mat.val[0])),
				const_cast<int *>(&mat.rowPtr[0]),
				const_cast<int *>(&mat.col[0]),
				perm,
				&nrhs,
				lu->_iparm,
				&msglvl,
				0,
				0,
				&error,
				lu->_dparm);

		return lu;
	}

	LU::~LU() {
		// Intel doc : Maximum number of factors with identical nonzero sparsity structure 
		// that must be keep at the same time in memory. 
		// In most applications this value is equal to 1.
		int maxfct = 1;

		// Intel doc: Indicates the actual matrix for the solution phase. 
		// With this scalar you can define which matrix to factorize.
		// In most applications this value is 1.
		int mnum = 1;

		int phase = -1; // Release all internal memory for all matrices

		// Intel doc: Holds the permutation vector of size n. 
		// You can use it to apply your own fill-in reducing ordering to the solver. 
		// The permutation vector perm is used by the solver if iparm(5) = 1.
		int *perm = 0;

		// Number of right-hand sides that need to be solved for.
		int nrhs = 0;

		int msglvl = 0; // Debug output => 1

		int error = 0;
		pardiso(const_cast<void **>(_pardisoHandle),
				&maxfct,
				&mnum,
				const_cast<int *>(&_mtype),
				&phase, 
				const_cast<int *>(&_n),
				const_cast<double *>(reinterpret_cast<const double *>(_val)),
				const_cast<int *>(_rowPtr),
				const_cast<int *>(_col),
				perm,
				&nrhs,
				const_cast<int *>(_iparm),
				&msglvl,
				0,
				0,
				&error,
				_dparm);
	}

	void LU::solve(std::complex<double> *bx) const
	{
		// Intel doc : Maximum number of factors with identical nonzero sparsity structure 
		// that must be keep at the same time in memory. 
		// In most applications this value is equal to 1.
		int maxfct = 1;

		// Intel doc: Indicates the actual matrix for the solution phase. 
		// With this scalar you can define which matrix to factorize.
		// In most applications this value is 1.
		int mnum = 1;

		int phase = 33; // Solve

		// Intel doc: Holds the permutation vector of size n. 
		// You can use it to apply your own fill-in reducing ordering to the solver. 
		// The permutation vector perm is used by the solver if iparm(5) = 1.
		int *perm = 0;

		// Number of right-hand sides that need to be solved for.
		int nrhs = 1;

		int msglvl = 0; // Debug output

		int error = 0;

		_iparm[5] = 1; // in-place

		std::vector<std::complex<double> > x(_n);

		pardiso(const_cast<void **>(_pardisoHandle),
				&maxfct,
				&mnum,
				const_cast<int *>(&_mtype),
				&phase, 
				const_cast<int *>(&_n),
				const_cast<double *>(reinterpret_cast<const double *>(_val)),
				const_cast<int *>(_rowPtr),
				const_cast<int *>(_col),
				perm,
				&nrhs,
				_iparm,
				&msglvl,
				(double *)bx,
				(double *)&x[0],
				&error,
				_dparm);
	}

	void LU::solve(const std::vector<std::complex<double> > &b, std::complex<double> *x) const {
		// Intel doc : Maximum number of factors with identical nonzero sparsity structure 
		// that must be keep at the same time in memory. 
		// In most applications this value is equal to 1.
		int maxfct = 1;

		// Intel doc: Indicates the actual matrix for the solution phase. 
		// With this scalar you can define which matrix to factorize.
		// In most applications this value is 1.
		int mnum = 1;

		int phase = 33; // Solve

		// Intel doc: Holds the permutation vector of size n. 
		// You can use it to apply your own fill-in reducing ordering to the solver. 
		// The permutation vector perm is used by the solver if iparm(5) = 1.
		int *perm = 0;

		// Number of right-hand sides that need to be solved for.
		int nrhs = 1;

		int msglvl = 0; // Debug output

		int error = 0;

		_iparm[5] = 0;

		pardiso(const_cast<void **>(_pardisoHandle),
				&maxfct,
				&mnum,
				const_cast<int *>(&_mtype),
				&phase, 
				const_cast<int *>(&_n),
				const_cast<double *>(reinterpret_cast<const double *>(_val)),
				const_cast<int *>(_rowPtr),
				const_cast<int *>(_col),
				perm,
				&nrhs,
				_iparm,
				&msglvl,
				(double *)&b[0],
				(double *)x,
				&error,
				_dparm);
	}
}
