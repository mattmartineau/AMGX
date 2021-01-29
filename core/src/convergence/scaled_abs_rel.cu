/* Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "amg_config.h"
#include "convergence/scaled_abs_rel.h"
#include "thrust_wrapper.h"

#include <algorithm>

namespace amgx
{
template<class IndexType, class ValueTypeA, class ValueTypeB>
__global__ void scaled_norm_factor_calc(
    int localNRows, ValueTypeA* Avals, IndexType* Arows, ValueTypeB* Ax, ValueTypeB* b, ValueTypeB x_avg, ValueTypeB* local_norm_factor)
{
    ValueTypeB normFactor = 0.0;

    for (int r = threadIdx.x + blockIdx.x * blockDim.x; r < localNRows; r += gridDim.x * blockDim.x)
    {
        ValueTypeA Arow_sum = amgx::types::util<ValueTypeB>::get_zero();

        // Read in the row
        for (int i = Arows[r]; i < Arows[r + 1]; ++i)
        {
            Arow_sum = Arow_sum + Avals[i];
        }

        normFactor += fabs(Ax[r] - Arow_sum * x_avg) + fabs(b[r] - Arow_sum * x_avg);
    }

    atomicAdd(&local_norm_factor[blockIdx.x], normFactor);
}


template<typename TConfig>
void ScaledAbsRelConvergence<TConfig>::compute_norm_factor(Matrix<TConfig>& A, Vector<TConfig>& b, Vector<TConfig>& x)
{
    nvtxRange cnf(__func__);

    int localNRows = A.get_num_rows();

    // Calculate Ax
    Vector<TConfig> Ax(localNRows);
    A.apply(x, Ax);

    // Calculate global average x
    ValueTypeB x_avg = thrust_wrapper::reduce(x.begin(), x.begin() + localNRows);
    A.manager->global_reduce_sum(&x_avg);
    x_avg /= (double)A.manager->num_rows_global;

    // Make a copy of b
    Vector<TConfig> b_cp(b);

    // Calculate row sums then the local norm factors
    constexpr int nthreads = 128;
    constexpr int nblocks = 8192;
    Vector<TConfig> local_norm_factor(nblocks, 0.0);
    scaled_norm_factor_calc<<<nblocks, nthreads>>>(
        localNRows,
        A.values.raw(),
        A.row_offsets.raw(),
        Ax.raw(),
        b_cp.raw(),
        x_avg,
        local_norm_factor.raw());

    // Reduce the normFactor over all ranks
    this->m_normFactor = thrust_wrapper::reduce(local_norm_factor.begin(), local_norm_factor.begin() + nblocks);

    A.manager->global_reduce_sum(&this->m_normFactor);

    // Print the norm factor
    std::stringstream info;
    info << "AmgX Scaled Norm Factor: " << this->m_normFactor << "\n";
    amgx_output(info.str().c_str(), info.str().length());
}

template<typename TConfig>
ScaledAbsRelConvergence<TConfig>::ScaledAbsRelConvergence(AMG_Config &cfg, const std::string &cfg_scope) : Convergence<TConfig>(cfg, cfg_scope)
{
}

template<class TConfig>
void ScaledAbsRelConvergence<TConfig>::convergence_init(Matrix<TConfig>& A, Vector<TConfig>& b, Vector<TConfig>& x)
{
    this->setTolerance(this->m_cfg->template getParameter<double>("tolerance", this->m_cfg_scope));
    this->m_alt_rel_tolerance = this->m_cfg->template getParameter<double>("alt_rel_tolerance", this->m_cfg_scope);

    // Calculate the norm factor for scaling the norm
    compute_norm_factor(A, b, x);
}

template<class TConfig>
bool ScaledAbsRelConvergence<TConfig>::convergence_update_and_check(const PODVec_h &nrm, const PODVec_h &nrm_ini)
{
    PODVec_h scaled_nrm(nrm);

    // Scale the norm by the provided norm factor
    for (int i = 0; i < nrm.size(); ++i)
    {
        scaled_nrm[i] = scaled_nrm[i] / this->m_normFactor;
    }

    bool res_converged = true;
    bool res_converged_abs = true;
    bool res_converged_abs_precision = true;

    for (int i = 0; i < scaled_nrm.size(); i++)
    {
        bool conv_abs = scaled_nrm[i] < this->m_tolerance;
        res_converged_abs = res_converged_abs && conv_abs;
        bool conv = (scaled_nrm[i] / scaled_nrm[i] <= this->m_alt_rel_tolerance);
        res_converged = res_converged && conv;
        bool conv_abs_precision = (scaled_nrm[i] <= std::max(scaled_nrm[i] * Epsilon_conv<ValueTypeB>::value(), (PODValueTypeB)(1e-20)));
        res_converged_abs_precision = res_converged_abs_precision && conv_abs_precision;
    }

    if (res_converged_abs_precision)
    {
        return true;
    }

    return res_converged || res_converged_abs;
}


/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class ScaledAbsRelConvergence<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // end namespace

