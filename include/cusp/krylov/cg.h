// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file cg.h
 *  \brief Conjugate Gradient (CG) method
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{
namespace krylov
{

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup krylov_methods Krylov Methods
 *  \ingroup iterative_solvers
 *  \{
 */

/*! \p cg : Conjugate Gradient method
 *
 * Solves the symmetric, positive-definite linear system A x = b
 * using the default convergence criteria.
 */
template <class LinearOperator,
          class Vector>
void cg(LinearOperator& A,
        Vector& x,
        Vector& b);

/*! \p cg : Conjugate Gradient method
 *
 * Solves the symmetric, positive-definite linear system A x = b without preconditioning.
 */
template <class LinearOperator,
          class Vector,
          class Monitor>
void cg(LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor);

/*! \p cg : Conjugate Gradient method
 *
 * Solves the symmetric, positive-definite linear system A x = b
 * with preconditioner \p M.
 *
 * \param A matrix of the linear system 
 * \param x approximate solution of the linear system
 * \param b right-hand side of the linear system
 * \param monitor montiors iteration and determines stopping conditions
 * \param M preconditioner for A
 *
 * \tparam LinearOperator is a matrix or subclass of \p linear_operator
 * \tparam Vector vector
 * \tparam Monitor is a monitor such as \p default_monitor or \p verbose_monitor
 * \tparam Preconditioner is a matrix or subclass of \p linear_operator
 *
 * \note \p A and \p M must be symmetric and positive-definite.
 *
 *  The following code snippet demonstrates how to use \p cg to 
 *  solve a 10x10 Poisson problem.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/monitor.h>
 *  #include <cusp/krylov/cg.h>
 *  #include <cusp/gallery/poisson.h>
 *  
 *  int main(void)
 *  {
 *      // create an empty sparse matrix structure (CSR format)
 *      cusp::csr_matrix<int, float, cusp::device_memory> A;
 *
 *      // initialize matrix
 *      cusp::gallery::poisson5pt(A, 10, 10);
 *
 *      // allocate storage for solution (x) and right hand side (b)
 *      cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
 *      cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);
 *
 *      // set stopping criteria:
 *      //  iteration_limit    = 100
 *      //  relative_tolerance = 1e-6
 *      cusp::verbose_monitor<float> monitor(b, 100, 1e-6);
 *
 *      // set preconditioner (identity)
 *      cusp::identity_operator<float, cusp::device_memory> M(A.num_rows, A.num_rows);
 *
 *      // solve the linear system A x = b
 *      cusp::krylov::cg(A, x, b, monitor, M);
 *
 *      return 0;
 *  }
 *  \endcode
 
 *  \see \p default_monitor
 *  \see \p verbose_monitor
 *
 */
template <class LinearOperator,
          class Vector,
          class Monitor,
          class Preconditioner>
void cg(LinearOperator& A,
        Vector& x,
        Vector& b,
        Monitor& monitor,
        Preconditioner& M);
/*! \}
 */

} // end namespace krylov
} // end namespace cusp

#include <cusp/krylov/detail/cg.inl>

