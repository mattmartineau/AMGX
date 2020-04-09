/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
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
#define COARSE_CLA_CONSO 0

#include <classical/classical_amg_level.h>
#include <amg_level.h>

#include <basic_types.h>
#include <cutil.h>
#include <multiply.h>
#include <transpose.h>
#include <truncate.h>
#include <blas.h>
#include <util.h>
#include <thrust/logical.h>
#include <thrust/remove.h>
#include <thrust/adjacent_difference.h>
#include <thrust_wrapper.h>

#include <thrust/extrema.h> // for minmax_element

#include <algorithm>
#include <assert.h>
#include <matrix_io.h>

#include <csr_multiply.h>

#include <thrust/logical.h>
#include <thrust/count.h>
#include <thrust/sort.h>

#include <profile.h>
#include <distributed/glue.h>
namespace amgx
{

namespace classical
{

void __global__ profiler_tag_1() {}
void __global__ profiler_tag_2() {}
void __global__ profiler_tag_3() {}

struct is_zero
{
    __host__ __device__
    bool operator()(const double &v)
    {
        return fabs(v) < 1e-10;
    }
};

#define AMGX_CAL_BLOCK_SIZE 256

/* There might be a situation where not all local_to_global_map columns are present in the matrix (because some rows were removed
   and the columns in these rows are therefore no longer present. This kernel creates the flags array that marks existing columns. */
template<typename ind_t>
__global__ __launch_bounds__( AMGX_CAL_BLOCK_SIZE )
void flag_existing_local_to_global_columns(ind_t n, ind_t *row_offsets, ind_t *col_indices, ind_t *flags)
{
    ind_t i, j, s, e, col;

    //go through the matrix
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
    {
        s = row_offsets[i];
        e = row_offsets[i + 1];

        for (j = s; j < e; j++)
        {
            col = col_indices[j];

            //flag columns outside of the square part (which correspond to local_to_global_map)
            if (col >= n)
            {
                flags[col - n] = 1;
            }
        }
    }
}

/* Renumber the indices based on the prefix-scan/sum of the flags array */
template<typename ind_t>
__global__ __launch_bounds__( AMGX_CAL_BLOCK_SIZE )
void compress_existing_local_columns(ind_t n, ind_t *row_offsets, ind_t *col_indices, ind_t *flags)
{
    ind_t i, j, s, e, col;

    //go through the matrix
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
    {
        s = row_offsets[i];
        e = row_offsets[i + 1];

        for (j = s; j < e; j++)
        {
            col = col_indices[j];

            //flag columns outside of the square part (which correspond to local_to_global_map)
            if (col >= n)
            {
                col_indices[j] = n + flags[col - n];
            }
        }
    }
}

/* compress the local to global columns indices based on the prefix-scan/sum of the flags array */
template<typename ind_t, typename ind64_t>
__global__ __launch_bounds__( AMGX_CAL_BLOCK_SIZE )
void compress_existing_local_to_global_columns(ind_t n, ind64_t *l2g_in, ind64_t *l2g_out, ind_t *flags)
{
    ind_t i;

    //go through the arrays (and copy the updated indices when needed)
    for (i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x)
    {
        if (flags[i] != flags[i + 1])
        {
            l2g_out[flags[i]] = l2g_in[i];
        }
    }
}


template <class T_Config>
Selector<T_Config> *chooseAggressiveSelector(AMG_Config *m_cfg, std::string std_scope)
{
    AMG_Config cfg;
    std::string cfg_string("");
    cfg_string += "default:";
    // if necessary, allocate aggressive selector + interpolator
    bool use_pmis = false, use_hmis = false;
    // default argument - use the same selector as normal coarsening
    std::string agg_selector = m_cfg->AMG_Config::getParameter<std::string>("aggressive_selector", std_scope);

    if (agg_selector == "DEFAULT")
    {
        std::string std_selector = m_cfg->AMG_Config::getParameter<std::string>("selector", std_scope);

        if      (std_selector == "PMIS") { cfg_string += "selector=AGGRESSIVE_PMIS"; use_pmis = true; }
        else if (std_selector == "HMIS") { cfg_string += "selector=AGGRESSIVE_HMIS"; use_hmis = true; }
        else
        {
            FatalError("Must use either PMIS or HMIS algorithms with aggressive coarsening", AMGX_ERR_NOT_IMPLEMENTED);
        }
    }
    // otherwise use specified selector
    else if (agg_selector == "PMIS") { cfg_string += "selector=AGGRESSIVE_PMIS"; use_pmis = true; }
    else if (agg_selector == "HMIS") { cfg_string += "selector=AGGRESSIVE_HMIS"; use_hmis = true; }
    else
    {
        FatalError("Invalid aggressive coarsener selected", AMGX_ERR_NOT_IMPLEMENTED);
    }

    // check a selector has been selected
    if (!use_pmis && !use_hmis)
    {
        FatalError("No aggressive selector chosen", AMGX_ERR_NOT_IMPLEMENTED);
    }

    cfg.parseParameterString(cfg_string.c_str());
    // now allocate the selector and interpolator
    return classical::SelectorFactory<T_Config>::allocate(cfg, "default" /*std_scope*/);
}

template <class T_Config>
Interpolator<T_Config> *chooseAggressiveInterpolator(AMG_Config *m_cfg, std::string std_scope)
{
    // temporary config and pointer to main config
    AMG_Config cfg;
    std::string cfg_string("");
    cfg_string += "default:";
    // Set the interpolator
    cfg_string += "interpolator=";
    cfg_string += m_cfg->AMG_Config::getParameter<std::string>("aggressive_interpolator", std_scope);
    cfg.parseParameterString(cfg_string.c_str());
    // now allocate the selector and interpolator
    return InterpolatorFactory<T_Config>::allocate(cfg, "default" /*std_scope*/);
}

template <class T_Config>
Classical_AMG_Level_Base<T_Config>::Classical_AMG_Level_Base(AMG_Class *amg) : AMG_Level<T_Config>(amg)
{
    strength = StrengthFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
    selector = classical::SelectorFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
    interpolator = InterpolatorFactory<T_Config>::allocate(*(amg->m_cfg), amg->m_cfg_scope);
    trunc_factor = amg->m_cfg->AMG_Config::getParameter<double>("interp_truncation_factor", amg->m_cfg_scope);
    max_elmts = amg->m_cfg->AMG_Config::getParameter<int>("interp_max_elements", amg->m_cfg_scope);
    max_row_sum = amg->m_cfg->AMG_Config::getParameter<double>("max_row_sum", amg->m_cfg_scope);
    num_aggressive_levels = amg->m_cfg->AMG_Config::getParameter<int>("aggressive_levels", amg->m_cfg_scope);
}

template <class T_Config>
Classical_AMG_Level_Base<T_Config>::~Classical_AMG_Level_Base()
{
    delete strength;
    delete selector;
    delete interpolator;
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::transfer_level(AMG_Level<TConfig1> *ref_lvl)
{
    Classical_AMG_Level_Base<TConfig1> *ref_cla_lvl = dynamic_cast<Classical_AMG_Level_Base<TConfig1>*>(ref_lvl);
    this->P.copy(ref_cla_lvl->P);
    this->R.copy(ref_cla_lvl->R);
    this->m_s_con.copy(ref_cla_lvl->m_s_con);
    this->m_scratch.copy(ref_cla_lvl->m_scratch);
    this->m_cf_map.copy(ref_cla_lvl->m_cf_map);
}

/****************************************
 * Computes the A, P, and R operators
 ***************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::createCoarseVertices()
{
    if (AMG_Level<T_Config>::getLevelIndex() < this->num_aggressive_levels)
    {
        if (selector) { delete selector; }

        selector = chooseAggressiveSelector<T_Config>(AMG_Level<T_Config>::amg->m_cfg, AMG_Level<T_Config>::amg->m_cfg_scope);
    }

    Matrix<T_Config> &RAP = this->getNextLevel( typename Matrix<T_Config>::memory_space( ) )->getA( );
    Matrix<T_Config> &A = this->getA();
    int size_all, size_full, nnz_full;

    if (!A.is_matrix_singleGPU())
    {
        int offset;
        // Need to get number of 2-ring rows
        A.getOffsetAndSizeForView(ALL, &offset, &size_all);
        A.getOffsetAndSizeForView(FULL, &offset, &size_full);
        A.getNnzForView(FULL, &nnz_full);
    }
    else
    {
        size_all = A.get_num_rows();
        size_full = A.get_num_rows();
        nnz_full = A.get_num_nz();
    }

    this->m_cf_map.resize(size_all);
    this->m_s_con.resize(nnz_full);
    this->m_scratch.resize(size_full);
    thrust::fill(this->m_cf_map.begin(), this->m_cf_map.end(), 0);
    cudaCheckError();
    thrust::fill(this->m_s_con.begin(), this->m_s_con.end(), false);
    cudaCheckError();
    thrust::fill(this->m_scratch.begin(), this->m_scratch.end(), 0);
    cudaCheckError();
    markCoarseFinePoints();
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::createCoarseMatrices()
{
    if (this->A->is_matrix_distributed() && this->A->manager->get_num_partitions() > 1)
    {
        createCoarseMatricesFlattened();
    }
    else
    {
        // allocate aggressive interpolator if needed
        if (AMG_Level<T_Config>::getLevelIndex() < this->num_aggressive_levels)
        {
            if (interpolator) { delete interpolator; }

            interpolator = chooseAggressiveInterpolator<T_Config>(AMG_Level<T_Config>::amg->m_cfg, AMG_Level<T_Config>::amg->m_cfg_scope);
        }

        Matrix<T_Config> &RAP = this->getNextLevel( typename Matrix<T_Config>::memory_space( ) )->getA( );
        Matrix<T_Config> &A = this->getA();
        /* WARNING: exit if D1 interpolator is selected in distributed setting */
        std::string s("");
        s += AMG_Level<T_Config>::amg->m_cfg->AMG_Config::getParameter<std::string>("interpolator", AMG_Level<T_Config>::amg->m_cfg_scope);

        if (A.is_matrix_distributed() && (s.compare("D1") == 0))
        {
            FatalError("D1 interpolation is not supported in distributed settings", AMGX_ERR_NOT_IMPLEMENTED);
        }

        /* WARNING: do not recompute prolongation (P) and restriction (R) when you
                    are reusing the level structure (structure_reuse_levels > 0) */
        if (this->isReuseLevel() == false)
        {
            computeProlongationOperator();
        }

        // Compute Restriction operator and coarse matrix Ac
        if (!this->A->is_matrix_distributed() || this->A->manager->get_num_partitions() == 1)
        {
            /* WARNING: see above warning. */
            if (this->isReuseLevel() == false)
            {
                computeRestrictionOperator();
            }

            computeAOperator();
        }
        else
        {
            /* WARNING: notice that in this case the computeRestructionOperator() is called
                        inside computeAOperator_distributed() routine. */
            computeAOperator_distributed();
        }

    // we also need to renumber columns of P and rows or R correspondingly since we changed RAP halo columns
    // for R we just keep track of renumbering in and exchange proper vectors in restriction
    // for P we actually need to renumber columns for prolongation:
        if (A.is_matrix_distributed() && this->A->manager->get_num_partitions() > 1)
        {
            RAP.set_initialized(0);
            // Renumber the owned nodes as interior and boundary (renumber rows and columns)
            // We are passing reuse flag to not create neighbours list from scratch, but rather update based on new halos
            RAP.manager->renumberMatrixOneRing(this->isReuseLevel());
            // Renumber the column indices of P and shuffle rows of P
            RAP.manager->renumber_P_R(this->P, this->R, A);
            // Create the B2L_maps for RAP
            {
                nvtxRange fdafds("createOneRingHaloRows RAP");
                RAP.manager->createOneRingHaloRows();
            }
            RAP.manager->getComms()->set_neighbors(RAP.manager->num_neighbors());
            RAP.setView(OWNED);
            RAP.set_initialized(1);
            // update # of columns in P - this is necessary for correct CSR multiply
            P.set_initialized(0);
            int new_num_cols = thrust_wrapper::reduce(P.col_indices.begin(), P.col_indices.end(), int(0), thrust::maximum<int>()) + 1;
            cudaCheckError();
            P.set_num_cols(new_num_cols);
            P.set_initialized(1);
        }

        RAP.copyAuxData(&A);

        if (!A.is_matrix_singleGPU() && RAP.manager == NULL)
        {
            RAP.manager = new DistributedManager<TConfig>();
        }

        if (this->getA().is_matrix_singleGPU())
        {
            this->m_next_level_size = this->getNextLevel(typename Matrix<TConfig>::memory_space() )->getA().get_num_rows() * this->getNextLevel(typename Matrix<TConfig>::memory_space() )->getA().get_block_dimy();
        }
        else
        {
            // m_next_level_size is the size that will be used to allocate xc, bc vectors
            int size, offset;
            this->getNextLevel(typename Matrix<TConfig>::memory_space())->getA().getOffsetAndSizeForView(FULL, &offset, &size);
            this->m_next_level_size = size * this->getNextLevel(typename Matrix<TConfig>::memory_space() )->getA().get_block_dimy();
        }
    }
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::markCoarseFinePoints()
{
    Matrix<T_Config> &A = this->getA();
    //allocate necessary memory
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    FVector weights;

    if (!A.is_matrix_singleGPU())
    {
        int size, offset;
        A.getOffsetAndSizeForView(FULL, &offset, &size);
        // size should now contain the number of 1-ring rows
        weights.resize(size);
    }
    else
    {
        weights.resize(A.get_num_rows());
    }

    thrust::fill(weights.begin(), weights.end(), 0.0);
    cudaCheckError();

    // extend A to include 1st ring nodes
    // compute strong connections and weights
    if (!A.is_matrix_singleGPU())
    {
        ViewType oldView = A.currentView();
        A.setView(FULL);
        strength->computeStrongConnectionsAndWeights(A, this->m_s_con, weights, this->max_row_sum);
        A.setView(oldView);
    }
    else
    {
        strength->computeStrongConnectionsAndWeights(A, this->m_s_con, weights, this->max_row_sum);
    }

    // Exchange the one-ring of the weights
    if (!A.is_matrix_singleGPU())
    {
        A.manager->exchange_halo(weights, weights.tag);
    }

    //mark coarse and fine points
    selector->markCoarseFinePoints(A, weights, this->m_s_con, this->m_cf_map, this->m_scratch);
    // we do resize cf_map to zero later, so we are saving separate copy
    this->m_cf_map.dirtybit = 1;

    // Do a two ring exchange of cf_map
    if (!A.is_matrix_singleGPU())
    {
        A.manager->exchange_halo_2ring(this->m_cf_map, m_cf_map.tag);
    }

    // Modify cf_map array such that coarse points are assigned a local index, while fine points entries are not touched
    selector->renumberAndCountCoarsePoints(this->m_cf_map, this->m_num_coarse_vertices, A.get_num_rows());
}


template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeProlongationOperator()
{
    this->Profile.tic("computeP");
    Matrix<T_Config> &A = this->getA();
    //allocate necessary memory
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecInt>::Type> IVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    //generate the interpolation matrix
    interpolator->generateInterpolationMatrix(A, this->m_cf_map, this->m_s_con, this->m_scratch, P, AMG_Level<TConfig>::amg);
    this->m_cf_map.clear();
    this->m_cf_map.shrink_to_fit();
    this->m_scratch.clear();
    this->m_scratch.shrink_to_fit();
    this->m_s_con.clear();
    this->m_s_con.shrink_to_fit();
    profileSubphaseTruncateP();

    // truncate based on max # of elements if desired
    if (this->max_elmts > 0 && P.get_num_rows() > 0)
    {
        Truncate<TConfig>::truncateByMaxElements(P, this->max_elmts);
    }

    if (this->m_min_rows_latency_hiding < 0 || P.get_num_rows() < this->m_min_rows_latency_hiding)
    {
        // This will cause bsrmv_with_mask to not do latency hiding
        P.setInteriorView(OWNED);
        P.setExteriorView(OWNED);
    }

    profileSubphaseNone();
    this->Profile.toc("computeP");
}

/**********************************************
 * computes R=P^T
 **********************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeRestrictionOperator()
{
    this->Profile.tic("computeR");
    R.set_initialized(0);
    P.setView(OWNED);
    transpose(P, R, P.get_num_rows());

    if (this->m_min_rows_latency_hiding < 0 || R.get_num_rows() < this->m_min_rows_latency_hiding)
    {
        // This will cause bsrmv_with_mask_restriction to not do latency hiding
        R.setInteriorView(OWNED);
        R.setExteriorView(OWNED);
    }

    R.set_initialized(1);
    this->Profile.toc("computeR");
}

/**********************************************
 * computes the Galerkin product: A_c=R*A*P
 **********************************************/

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1()
{
    this->Profile.tic("computeA");
    Matrix<TConfig_h> RA;
    RA.addProps(CSR);
    RA.set_block_dimx(this->getA().get_block_dimx());
    RA.set_block_dimy(this->getA().get_block_dimy());
    Matrix<TConfig_h> &RAP = this->getNextLevel( typename Matrix<TConfig_h>::memory_space( ) )->getA( );
    RAP.addProps(CSR);
    RAP.set_block_dimx(this->getA().get_block_dimx());
    RAP.set_block_dimy(this->getA().get_block_dimy());
    Matrix<TConfig_h> &Atmp = this->getA();
    multiplyMM(this->R, this->getA(), RA);
    multiplyMM(RA, this->P, RAP);
    RAP.sortByRowAndColumn();
    RAP.set_initialized(1);
    this->Profile.toc("computeA");
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1_distributed()
{
    FatalError("Distributed classical AMG not implemented for host\n", AMGX_ERR_NOT_IMPLEMENTED);
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1()
{
    this->Profile.tic("computeA");
    Matrix<TConfig_d> &RAP = this->getNextLevel( device_memory( ) )->getA( );
    RAP.addProps(CSR);
    RAP.set_block_dimx(this->getA().get_block_dimx());
    RAP.set_block_dimy(this->getA().get_block_dimy());
    this->R.set_initialized( 0 );
    this->R.addProps( CSR );
    this->R.set_initialized( 1 );
    this->P.set_initialized( 0 );
    this->P.addProps( CSR );
    this->P.set_initialized( 1 );
    void *wk = AMG_Level<TConfig_d>::amg->getCsrWorkspace();

    if ( wk == NULL )
    {
        wk = CSR_Multiply<TConfig_d>::csr_workspace_create( *(AMG_Level<TConfig_d>::amg->m_cfg), AMG_Level<TConfig_d>::amg->m_cfg_scope );
        AMG_Level<TConfig_d>::amg->setCsrWorkspace( wk );
    }

    int spmm_verbose = this->amg->m_cfg->AMG_Config::getParameter<int>("spmm_verbose", this->amg->m_cfg_scope);

    if ( spmm_verbose )
    {
        typedef typename Matrix<TConfig_d>::IVector::const_iterator Iterator;
        typedef thrust::pair<Iterator, Iterator> Result;
        std::ostringstream buffer;
        buffer << "SPMM: Level " << this->getLevelIndex() << std::endl;

        if ( this->getLevelIndex() == 0 )
        {
            device_vector_alloc<int> num_nz( this->getA().row_offsets.size() );
            thrust::adjacent_difference( this->getA().row_offsets.begin(), this->getA().row_offsets.end(), num_nz.begin() );
            cudaCheckError();
            Result result = thrust::minmax_element( num_nz.begin() + 1, num_nz.end() );
            cudaCheckError();
            int min_size = *result.first;
            int max_size = *result.second;
            int sum = thrust_wrapper::reduce( num_nz.begin() + 1, num_nz.end() );
            cudaCheckError();
            double avg_size = double(sum) / this->getA().get_num_rows();
            buffer << "SPMM: A: " << std::endl;
            buffer << "SPMM: Matrix avg row size: " << avg_size << std::endl;
            buffer << "SPMM: Matrix min row size: " << min_size << std::endl;
            buffer << "SPMM: Matrix max row size: " << max_size << std::endl;
        }

        device_vector_alloc<int> num_nz( this->P.row_offsets.size() );
        thrust::adjacent_difference( this->P.row_offsets.begin(), this->P.row_offsets.end(), num_nz.begin() );
        cudaCheckError();
        Result result = thrust::minmax_element( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        int min_size = *result.first;
        int max_size = *result.second;
        int sum = thrust_wrapper::reduce( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        double avg_size = double(sum) / this->P.get_num_rows();
        buffer << "SPMM: P: " << std::endl;
        buffer << "SPMM: Matrix avg row size: " << avg_size << std::endl;
        buffer << "SPMM: Matrix min row size: " << min_size << std::endl;
        buffer << "SPMM: Matrix max row size: " << max_size << std::endl;
        num_nz.resize( this->R.row_offsets.size() );
        thrust::adjacent_difference( this->R.row_offsets.begin(), this->R.row_offsets.end(), num_nz.begin() );
        cudaCheckError();
        result = thrust::minmax_element( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        min_size = *result.first;
        max_size = *result.second;
        sum = thrust_wrapper::reduce( num_nz.begin() + 1, num_nz.end() );
        cudaCheckError();
        avg_size = double(sum) / this->R.get_num_rows();
        buffer << "SPMM: R: " << std::endl;
        buffer << "SPMM: Matrix avg row size: " << avg_size << std::endl;
        buffer << "SPMM: Matrix min row size: " << min_size << std::endl;
        buffer << "SPMM: Matrix max row size: " << max_size << std::endl;
        amgx_output( buffer.str().c_str(), static_cast<int>( buffer.str().length() ) );
    }

    RAP.set_initialized( 0 );
    CSR_Multiply<TConfig_d>::csr_galerkin_product( this->R, this->getA(), this->P, RAP, NULL, NULL, NULL, NULL, NULL, NULL, wk );
    RAP.set_initialized( 1 );
    int spmm_no_sort = this->amg->m_cfg->AMG_Config::getParameter<int>("spmm_no_sort", this->amg->m_cfg_scope);
    this->Profile.toc("computeA");
}
/**********************************************
 * computes the restriction: rr=R*r
 **********************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::restrictResidual(VVector &r, VVector &rr)
{
// we need to resize residual vector to make sure it can store halo rows to be sent
    if (!P.is_matrix_singleGPU())
    {
        typedef typename TConfig::MemSpace MemorySpace;
        Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
#if COARSE_CLA_CONSO
        int desired_size ;

        if (this->getNextLevel(MemorySpace())->isConsolidationLevel())
        {
            desired_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets_before_glue[Ac.manager->neighbors_before_glue.size()] * rr.get_block_size());
        }
        else
        {
            desired_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()] * rr.get_block_size());
        }

#else
        int desired_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()] * rr.get_block_size());
#endif
        rr.resize(desired_size);
    }

#if 1
    this->Profile.tic("restrictRes");

    // Disable speculative send of rr
    if (P.is_matrix_singleGPU())
    {
        multiply( R, r, rr);
    }
    else
    {
        multiply_with_mask_restriction( R, r, rr, P);
    }

#endif
    // exchange halo residuals & add residual contribution from neighbors
    rr.dirtybit = 1;

    if (!P.is_matrix_singleGPU())
    {
        int desired_size = P.manager->halo_offsets[P.manager->neighbors.size()] * rr.get_block_size();

        if (rr.size() < desired_size)
        {
            rr.resize(desired_size);
        }
    }

    this->Profile.toc("restrictRes");
}

struct is_minus_one
{
    __host__ __device__
    bool operator()(const int &x)
    {
        return x == -1;
    }
};


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::computeAOperator_1x1_distributed()
{
    Matrix<TConfig_d> &A = this->getA();
    Matrix<TConfig_d> &P = this->P;
    Matrix<TConfig_d> &RAP = this->getNextLevel( device_memory( ) )->getA( );
    RAP.addProps(CSR);
    RAP.set_block_dimx(this->getA().get_block_dimx());
    RAP.set_block_dimy(this->getA().get_block_dimy());
    IndexType num_parts = A.manager->get_num_partitions();
    IndexType num_neighbors = A.manager->num_neighbors();
    IndexType my_rank = A.manager->global_id();
    // OWNED includes interior and boundary
    A.setView(OWNED);
    int num_owned_coarse_pts = P.manager->halo_offsets[0];
    int num_owned_fine_pts = A.manager->halo_offsets[0];

    // Initialize RAP.manager
    if (RAP.manager == NULL)
    {
        RAP.manager = new DistributedManager<TConfig_d>();
    }

    RAP.manager->A = &RAP;
    RAP.manager->setComms(A.manager->getComms());
    RAP.manager->set_global_id(my_rank);
    RAP.manager->set_num_partitions(num_parts);
    RAP.manager->part_offsets_h = P.manager->part_offsets_h;
    RAP.manager->part_offsets = P.manager->part_offsets;
    RAP.manager->set_base_index(RAP.manager->part_offsets_h[my_rank]);
    RAP.manager->set_index_range(num_owned_coarse_pts);
    RAP.manager->num_rows_global = RAP.manager->part_offsets_h[num_parts];
    // --------------------------------------------------------------------
    // Using the B2L_maps of matrix A, identify the rows of P that need to be sent to neighbors,
    // so that they can compute A*P
    // Once rows of P are identified, convert the column indices to global indices, and send them to neighbors
    //  ---------------------------------------------------------------------------
    // Copy some information about the manager of P, since we don't want to modify those
    IVector_h P_neighbors = P.manager->neighbors;
    I64Vector_h P_halo_ranges_h = P.manager->halo_ranges_h;
    I64Vector_d P_halo_ranges = P.manager->halo_ranges;
    RAP.manager->local_to_global_map = P.manager->local_to_global_map;
    IVector_h P_halo_offsets = P.manager->halo_offsets;
    // Create a temporary distributed arranger
    DistributedArranger<TConfig_d> *prep = new DistributedArranger<TConfig_d>;
    prep->exchange_halo_rows_P(A, this->P, RAP.manager->local_to_global_map, P_neighbors, P_halo_ranges_h, P_halo_ranges, P_halo_offsets, RAP.manager->part_offsets_h, RAP.manager->part_offsets, num_owned_coarse_pts, RAP.manager->part_offsets_h[my_rank]);
    cudaCheckError();
    // At this point, we can compute RAP_full which contains some rows that will need to be sent to neighbors
    // i.e. RAP_full = [ RAP_int ]
    //                 [ RAP_ext ]
    // RAP is [ RAP_int ] + [RAP_ext_received_from_neighbors]
    // We can reuse the serial galerkin product since R, A and P use local indices
    // TODO: latency hiding (i.e. compute RAP_ext, exchange_matrix_halo, then do RAP_int)
    /* WARNING: do not recompute prolongation (P) and restriction (R) when you
                are reusing the level structure (structure_reuse_levels > 0) */
    /* We force for matrix P to have only owned rows to be seen for the correct galerkin product computation*/
    this->P.set_initialized(0);
    this->P.set_num_rows(num_owned_fine_pts);
    this->P.addProps( CSR );
    this->P.set_initialized(1);

    if (this->isReuseLevel() == false)
    {
        this->R.set_initialized( 0 );
        this->R.addProps( CSR );
        // Take the tranpose of P to get R
        // Single-GPU transpose, no mpi exchange
        this->computeRestrictionOperator();
        this->R.set_initialized( 1 );
    }

    this->Profile.tic("computeA");
    Matrix<TConfig_d> RAP_full;
    // Initialize the workspace needed for galerkin product
    void *wk = AMG_Level<TConfig_d>::amg->getCsrWorkspace();

    if ( wk == NULL )
    {
        wk = CSR_Multiply<TConfig_d>::csr_workspace_create( *(AMG_Level<TConfig_d>::amg->m_cfg), AMG_Level<TConfig_d>::amg->m_cfg_scope );
        AMG_Level<TConfig_d>::amg->setCsrWorkspace( wk );
    }

    // Single-GPU RAP, no mpi exchange
    RAP_full.set_initialized( 0 );
    /* WARNING: Since A is reordered (into interior and boundary nodes), while R and P are not reordered,
                you must unreorder A when performing R*A*P product in ordre to obtain the correct result. */
    CSR_Multiply<TConfig_d>::csr_galerkin_product( this->R, this->getA(), this->P, RAP_full,
            /* permutation for rows of R, A and P */       NULL, NULL /*&(this->getA().manager->renumbering)*/,        NULL,
            /* permutation for cols of R, A and P */       NULL, NULL /*&(this->getA().manager->inverse_renumbering)*/, NULL,
            wk );
    RAP_full.set_initialized( 1 );
    this->Profile.toc("computeA");
    // ----------------------------------------------------------------------------------------------
    // Now, send rows of RAP_full requireq by neighbors, received rows from neighbors and create RAP
    // ----------------------------------------------------------------------------------------------
    prep->exchange_RAP_ext(RAP, RAP_full, A, this->P, P_halo_offsets, RAP.manager->local_to_global_map, P_neighbors, P_halo_ranges_h, P_halo_ranges, RAP.manager->part_offsets_h, RAP.manager->part_offsets, num_owned_coarse_pts, RAP.manager->part_offsets_h[my_rank], wk);
    // Delete temporary distributed arranger
    delete prep;
    /* WARNING: The RAP matrix generated at this point contains extra rows (that correspond to rows of R,
       that was obtained by locally transposing P). This rows are ignored by setting the # of matrix rows
       to be smaller, so that they correspond to number of owned coarse nodes. This should be fine, but
       it leaves holes in the matrix as there might be columns that belong to the extra rows that now do not
       belong to the smaller matrix with number of owned coarse nodes rows. The same is trued about the
       local_to_global_map. These two data structures match at this point. However, in the next calls
       local_to_global (exclusively) will be used to geberate B2L_maps (wihtout going through column indices)
       which creates extra elements in the B2L that simply do not exist in the new matrices. I strongly suspect
       this is the reason fore the bug. The below fix simply compresses the matrix so that there are no holes
       in it, or in the local_2_global_map. */
    //mark local_to_global_columns that exist in the owned coarse nodes rows.
    IndexType nrow = RAP.get_num_rows();
    IndexType ncol = RAP.get_num_cols();
    IndexType nl2g = ncol - nrow;

    if (nl2g > 0)
    {
        IVector   l2g_p(nl2g + 1, 0); //+1 is needed for prefix_sum/exclusive_scan
        I64Vector l2g_t(nl2g, 0);
        IndexType nblocks = (nrow + AMGX_CAL_BLOCK_SIZE - 1) / AMGX_CAL_BLOCK_SIZE;

        if (nblocks > 0)
            flag_existing_local_to_global_columns<int> <<< nblocks, AMGX_CAL_BLOCK_SIZE>>>
            (nrow, RAP.row_offsets.raw(), RAP.col_indices.raw(), l2g_p.raw());

        cudaCheckError();
        /*
        //slow version of the above kernel
        for(int ii=0; ii<nrow; ii++){
            int s = RAP.row_offsets[ii];
            int e = RAP.row_offsets[ii+1];
            for (int jj=s; jj<e; jj++) {
                int col = RAP.col_indices[jj];
                if (col>=nrow){
                    int kk = col-RAP.get_num_rows();
                    l2g_p[kk] = 1;
                }
            }
        }
        cudaCheckError();
        */
        //create a pointer map for their location using prefix sum
        thrust_wrapper::exclusive_scan(l2g_p.begin(), l2g_p.end(), l2g_p.begin());
        int new_nl2g = l2g_p[nl2g];

        //compress the columns using the pointer map
        if (nblocks > 0)
            compress_existing_local_columns<int> <<< nblocks, AMGX_CAL_BLOCK_SIZE>>>
            (nrow, RAP.row_offsets.raw(), RAP.col_indices.raw(), l2g_p.raw());

        cudaCheckError();
        /*
        //slow version of the above kernel
        for(int ii=0; ii<nrow; ii++){
            int s = RAP.row_offsets[ii];
            int e = RAP.row_offsets[ii+1];
            for (int jj=s; jj<e; jj++) {
                int col = RAP.col_indices[jj];
                if (col>=nrow){
                    int kk = col-RAP.get_num_rows();
                    RAP.col_indices[jj] = nrow+l2g_p[kk];
                }
            }
        }
        cudaCheckError();
        */
        //adjust matrix size (number of columns) accordingly
        RAP.set_initialized(0);
        RAP.set_num_cols(nrow + new_nl2g);
        RAP.set_initialized(1);
        //compress local_to_global_map using the pointer map
        nblocks = (nl2g + AMGX_CAL_BLOCK_SIZE - 1) / AMGX_CAL_BLOCK_SIZE;

        if (nblocks > 0)
            compress_existing_local_to_global_columns<int, int64_t> <<< nblocks, AMGX_CAL_BLOCK_SIZE>>>
            (nl2g, RAP.manager->local_to_global_map.raw(), l2g_t.raw(), l2g_p.raw());

        cudaCheckError();
        thrust::copy(l2g_t.begin(), l2g_t.begin() + new_nl2g, RAP.manager->local_to_global_map.begin());
        cudaCheckError();
        /*
        //slow version of the above kernel (through Thrust)
        for(int ii=0; ii<(l2g_p.size()-1); ii++){
            if (l2g_p[ii] != l2g_p[ii+1]){
                RAP.manager->local_to_global_map[l2g_p[ii]] = RAP.manager->local_to_global_map[ii];
            }
        }
        cudaCheckError();
        */
        //adjust local_to_global_map size accordingly
        RAP.manager->local_to_global_map.resize(new_nl2g);
    }
}

/**********************************************
 * prolongates the error: x+=P*e
 **********************************************/
template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::prolongateAndApplyCorrection(VVector &e, VVector &bc, VVector &x, VVector &tmp)
{
    this->Profile.tic("proCorr");
    // Use P.manager to exchange halo of e before doing P
    // (since P has columns belonging to one of P.neighbors)
    e.dirtybit = 1;

    if (!P.is_matrix_singleGPU())
    {
        // get coarse matrix
        typedef typename TConfig::MemSpace MemorySpace;
        Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
#if COARSE_CLA_CONSO
        int e_size;

        if (this->getNextLevel(MemorySpace())->isConsolidationLevel())
        {
            e_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets_before_glue[Ac.manager->neighbors_before_glue.size()]) * e.get_block_size();
        }
        else
        {
            e_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()]) * e.get_block_size();
        }

        if (e.size() < e_size) { e.resize(e_size); }

#else
        int e_size = std::max(P.manager->halo_offsets[P.manager->neighbors.size()], Ac.manager->halo_offsets[Ac.manager->neighbors.size()]) * e.get_block_size();
        e.resize(e_size);
#endif
    }

    if (P.is_matrix_singleGPU())
    {
        if (e.size() > 0)
        {
            multiply( P, e, tmp);
        }
    }
    else
    {
        multiply_with_mask( P, e, tmp);
    }

    // get owned num rows for fine matrix
    int owned_size;

    if (this->A->is_matrix_distributed())
    {
        int owned_offset;
        P.manager->getOffsetAndSizeForView(OWNED, &owned_offset, &owned_size);
    }
    else
    {
        owned_size = x.size();
    }

    //apply
    axpby(x, tmp, x, ValueType(1), ValueType(1), 0, owned_size);
    this->Profile.toc("proCorr");
    x.dirtybit = 1;
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeAOperator()
{
    if (this->A->get_block_size() == 1)
    {
        computeAOperator_1x1();
    }
    else
    {
        FatalError("Classical AMG not implemented for block_size != 1", AMGX_ERR_NOT_IMPLEMENTED);
    }
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::computeAOperator_distributed()
{
    if (this->A->get_block_size() == 1)
    {
        computeAOperator_1x1_distributed();
    }
    else
    {
        FatalError("Classical AMG not implemented for block_size != 1", AMGX_ERR_NOT_IMPLEMENTED);
    }
}


template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::consolidateVector(VVector &x)
{
#ifdef AMGX_WITH_MPI
#if COARSE_CLA_CONSO
    typedef typename TConfig::MemSpace MemorySpace;
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
    MPI_Comm comm, temp_com;
    comm = Ac.manager->getComms()->get_mpi_comm();
    temp_com = compute_glue_matrices_communicator(Ac);
    glue_vector(Ac, comm, x, temp_com);
#endif
#endif
}

template <class T_Config>
void Classical_AMG_Level_Base<T_Config>::unconsolidateVector(VVector &x)
{
#ifdef AMGX_WITH_MPI
#if COARSE_CLA_CONSO
    typedef typename TConfig::MemSpace MemorySpace;
    Matrix<TConfig> &A = this->getA();
    Matrix<TConfig> &Ac = this->getNextLevel( MemorySpace( ) )->getA();
    MPI_Comm comm, temp_com;
    comm = Ac.manager->getComms()->get_mpi_comm();
    temp_com = compute_glue_matrices_communicator(Ac);
    unglue_vector(Ac, comm, x, temp_com, x);
#endif
#endif
}











template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::createCoarseMatricesFlattened()
{
}

template <int coop, class T>
__global__ void export_matrix_elements_global_flat(INDEX_TYPE *row_offsets, T *values, INDEX_TYPE bsize, INDEX_TYPE *maps, INDEX_TYPE *pointers, T *output, INDEX_TYPE *col_indices, int64_t *output2, INDEX_TYPE size, int64_t *local_to_global, INDEX_TYPE *q, INDEX_TYPE num_owned_pts, int64_t base_index)
{
    int idx = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (idx < size)
    {
        int row = maps[idx];

        if (q != NULL)
        {
            row = q[row];
        }

        INDEX_TYPE src_base = row_offsets[row];
        INDEX_TYPE dst_base = pointers[idx];

        for (int m = coopIdx; m < row_offsets[row + 1]*bsize - src_base * bsize; m += coop)
        {
            output[dst_base * bsize + m] = values[src_base * bsize + m];
        }

        for (int m = coopIdx; m < row_offsets[row + 1] - src_base; m += coop)
        {
            int col = col_indices[src_base + m];

            if (col < num_owned_pts)
            {
                output2[dst_base + m] = (int64_t) col_indices[src_base + m] + base_index;
            }
            else
            {
                output2[dst_base + m] = local_to_global[col_indices[src_base + m] - num_owned_pts];
            }
        }

        idx += gridDim.x * blockDim.x / coop;
    }
}

__global__ void write_matrix_rowsize_flat(INDEX_TYPE *maps, INDEX_TYPE *row_offsets, INDEX_TYPE size, INDEX_TYPE *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < size)
    {
        int row = maps[idx];
        output[idx] = row_offsets[row + 1] - row_offsets[row];
        idx += gridDim.x * blockDim.x;
    }
}

template <int coop>
__global__ void calc_num_neighbors_v2_global_flat(INDEX_TYPE *row_offsets, int64_t *col_indices, int64_t *part_offsets, INDEX_TYPE *exists, INDEX_TYPE num_part, INDEX_TYPE my_id, INDEX_TYPE num_rows)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;
    int64_t start = part_offsets[my_id];
    int64_t end = part_offsets[my_id + 1];

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            int64_t col = col_indices[i];

            // Check if column point to halo node
            if (col < start || col >= end)
            {
                int part = 0;

                while (part < num_part && (col < part_offsets[part] || col >= part_offsets[part + 1]))
                {
                    part++;
                }

                if (part < num_part && (col >= part_offsets[part] && col < part_offsets[part + 1]))
                {
                    exists[part] = 1;
                }
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

template <int coop, int set_val>
__global__ void flag_halo_nodes_global_flat(INDEX_TYPE *row_offsets, INDEX_TYPE num_rows, int64_t *global_col_indices, int64_t *part_ranges, INDEX_TYPE *flags, INDEX_TYPE *flag_offsets, int64_t base, INDEX_TYPE range, INDEX_TYPE num_neighbors, INDEX_TYPE *local_col_indices)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            int64_t col = global_col_indices[i];

            if (col < base || col >= base + range)
            {
                // exterior node
                int part = 0;

                while (part < num_neighbors && (col < part_ranges[2 * part] || col >= part_ranges[2 * part + 1])) { part++; }

                if (part < num_neighbors && (col >= part_ranges[2 * part] && col < part_ranges[2 * part + 1]))
                {
                    // check if the flag is already set (i.e. 1-ring node) then skip it
                    int flag = flags[flag_offsets[part] + col - part_ranges[2 * part]];

                    if (flag == 0)
                    {
                        flags[flag_offsets[part] + col - part_ranges[2 * part]] = set_val;
                    }
                    else if (flag < 0)    // it is possible that flag is already set to 1 in this kernel, so if it's negative then it's the 1st ring
                    {
                        // update local index, note that local to global mapping is unchanged for 1st ring
                        local_col_indices[i] = -flags[flag_offsets[part] + col - part_ranges[2 * part]] - 1;
                    }
                }
            }
            else
            {
                // interior node
                local_col_indices[i] = col - base;
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

struct is_less_than_zero
{
    __host__ __device__
    bool operator()(int x) const
    {
        return x < 0;
    }
};

template <int coop>
__global__ void calc_new_halo_mapping_ring2_flat(INDEX_TYPE *row_offsets, INDEX_TYPE num_rows, int64_t *global_col_indices, int64_t *part_ranges, int64_t base, INDEX_TYPE range, INDEX_TYPE num_neighbors, INDEX_TYPE *halo_offsets, INDEX_TYPE *neighbor_offsets, INDEX_TYPE *neighbor_nodes, int64_t *local_to_global, INDEX_TYPE *local_col_indices)
{
    int row = blockIdx.x * blockDim.x / coop + threadIdx.x / coop;
    int coopIdx = threadIdx.x % coop;

    while (row < num_rows)
    {
        for (int i = row_offsets[row] + coopIdx; i < row_offsets[row + 1]; i += coop)
        {
            int64_t col = global_col_indices[i];

            if ((col < base || col >= base + range) && local_col_indices[i] == -1)
            {
                // update only 2nd ring halo indices here, 1st ring is updated already
                int part = 0;

                while (part < num_neighbors && (col < part_ranges[2 * part] || col >= part_ranges[2 * part + 1])) { part++; }

                if (part < num_neighbors && (col >= part_ranges[2 * part] && col < part_ranges[2 * part + 1]))
                {
                    int pos = col - part_ranges[2 * part];
                    local_col_indices[i] = halo_offsets[part] + neighbor_nodes[neighbor_offsets[part] + pos];
                    local_to_global[local_col_indices[i] - range] = col;
                }
            }
        }

        row += gridDim.x * blockDim.x / coop;
    }
}

__global__ void flag_halo_nodes_local_v2_flat(int64_t *part_ranges, INDEX_TYPE *flags, INDEX_TYPE *flag_offsets, int64_t base, INDEX_TYPE range, INDEX_TYPE num_part, INDEX_TYPE local_to_global_size, int64_t *local_to_global)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < local_to_global_size)
    {
        int64_t global_col = local_to_global[tid];
        int col =  range + tid;
        // Identify the partition that owns that node
        int part = 0;

        while (part < num_part && (global_col < part_ranges[2 * part] || global_col >= part_ranges[2 * part + 1])) { part++; }

        // Flag the corresponding node in flags array
        if (part < num_part && (global_col >= part_ranges[2 * part] && global_col < part_ranges[2 * part + 1]))
        {
            flags[flag_offsets[part] + global_col - part_ranges[2 * part]] = -col - 1;
        }

        tid += gridDim.x * blockDim.x;
    }
}

__global__ void populate_B2L_flat(INDEX_TYPE *indexing, INDEX_TYPE *output, INDEX_TYPE last, INDEX_TYPE size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    while (row < size)
    {
        if (row == size - 1)
        {
            if (last) { output[indexing[row]] = row; }
        }
        else if (indexing[row] != indexing[row + 1])
        {
            output[indexing[row]] = row;
        }

        row += gridDim.x * blockDim.x;
    }
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Classical_AMG_Level<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>>::createCoarseMatricesFlattened()
{
    nvtxRange nvtx_ccmf(__func__);

    Matrix<TConfig_d> &RAP = this->getNextLevel(typename Matrix<TConfig_d>::memory_space())->getA();
    Matrix<TConfig_d> &A = this->getA();
    Matrix<TConfig_d> &P = this->P;

    // allocate aggressive interpolator if needed
    if (AMG_Level<TConfig_d>::getLevelIndex() < this->num_aggressive_levels)
    {
        if (this->interpolator)
        {
            delete this->interpolator;
        }

        this->interpolator = chooseAggressiveInterpolator<TConfig_d>(AMG_Level<TConfig_d>::amg->m_cfg, AMG_Level<TConfig_d>::amg->m_cfg_scope);
    }

    /* WARNING: exit if D1 interpolator is selected in distributed setting */
    std::string s("");
    s += AMG_Level<TConfig_d>::amg->m_cfg->AMG_Config::getParameter<std::string>("interpolator", AMG_Level<TConfig_d>::amg->m_cfg_scope);

    if (A.is_matrix_distributed() && (s.compare("D1") == 0))
    {
        FatalError("D1 interpolation is not supported in distributed settings", AMGX_ERR_NOT_IMPLEMENTED);
    }

    /* WARNING: do not recompute prolongation (P) and restriction (R) when you
                are reusing the level structure (structure_reuse_levels > 0) */
    if (this->isReuseLevel() == false)
    {
        {
            nvtxRange nvtx_gim("generateInterpolationMatrix");

            //generate the interpolation matrix
            this->interpolator->generateInterpolationMatrix(A, this->m_cf_map, this->m_s_con, this->m_scratch, P, AMG_Level<TConfig_d>::amg);
        }

        this->m_cf_map.clear();
        this->m_cf_map.shrink_to_fit();
        this->m_scratch.clear();
        this->m_scratch.shrink_to_fit();
        this->m_s_con.clear();
        this->m_s_con.shrink_to_fit();

        // truncate based on max # of elements if desired
        if (this->max_elmts > 0 && P.get_num_rows() > 0)
        {
            Truncate<TConfig_d>::truncateByMaxElements(P, this->max_elmts);
        }

        if (this->m_min_rows_latency_hiding < 0 || P.get_num_rows() < this->m_min_rows_latency_hiding)
        {
            // This will cause bsrmv_with_mask to not do latency hiding
            P.setInteriorView(OWNED);
            P.setExteriorView(OWNED);
        }
    }

    RAP.addProps(CSR);
    RAP.set_block_dimx(this->getA().get_block_dimx());
    RAP.set_block_dimy(this->getA().get_block_dimy());
    IndexType num_parts = A.manager->get_num_partitions();
    IndexType num_neighbors = A.manager->num_neighbors();
    IndexType my_rank = A.manager->global_id();
    // OWNED includes interior and boundary
    A.setView(OWNED);
    int num_owned_coarse_pts = P.manager->halo_offsets[0];
    int num_owned_fine_pts = A.manager->halo_offsets[0];

    // Initialize RAP.manager
    if (RAP.manager == NULL)
    {
        RAP.manager = new DistributedManager<TConfig_d>();
    }

    RAP.manager->A = &RAP;
    RAP.manager->setComms(A.manager->getComms());
    RAP.manager->set_global_id(my_rank);
    RAP.manager->set_num_partitions(num_parts);
    RAP.manager->part_offsets_h = P.manager->part_offsets_h;
    RAP.manager->part_offsets = P.manager->part_offsets;
    RAP.manager->set_base_index(RAP.manager->part_offsets_h[my_rank]);
    RAP.manager->set_index_range(num_owned_coarse_pts);
    RAP.manager->num_rows_global = RAP.manager->part_offsets_h[num_parts];

    // --------------------------------------------------------------------
    // Using the B2L_maps of matrix A, identify the rows of P that need to be sent to neighbors,
    // so that they can compute A*P
    // Once rows of P are identified, convert the column indices to global indices, and send them to neighbors
    //  ---------------------------------------------------------------------------

    // Copy some information about the manager of P, since we don't want to modify those
    IVector_h P_neighbors = P.manager->neighbors;
    I64Vector_h P_halo_ranges_h = P.manager->halo_ranges_h;
    I64Vector_d P_halo_ranges = P.manager->halo_ranges;
    RAP.manager->local_to_global_map = P.manager->local_to_global_map;
    IVector_h P_halo_offsets = P.manager->halo_offsets;

    // Create a temporary distributed arranger
    DistributedArranger<TConfig_d> *prep = new DistributedArranger<TConfig_d>;

    int coarse_base_index = RAP.manager->part_offsets_h[my_rank];
    if (P.hasProps(DIAG) || P.get_block_size() != 1)
    {
        FatalError("P with external diagonal or block_size != 1 not supported", AMGX_ERR_NOT_IMPLEMENTED);
    }

    typedef typename Matrix<TConfig_d>::MVector MVector;
    std::vector<IVector> halo_rows_P_row_offsets;
    std::vector<I64Vector> halo_rows_P_col_indices;
    std::vector<IVector> halo_rows_P_local_col_indices;
    std::vector<MVector> halo_rows_P_values;

    {
        nvtxRange nvtx_copy_P("copy_P_halos");
        // In this function, store in halo_rows_P_row_offsets, halo_rows_P_col_indices and halo_rows_P_values, the rows of P that need to be sent to each neighbors
        // halo_rows_P_col_indices stores global indices

        halo_rows_P_row_offsets.resize(num_neighbors);
        halo_rows_P_col_indices.resize(num_neighbors);
        halo_rows_P_values.resize(num_neighbors);

        int num_rings_to_send = 1;

        //Scratch space computation
        int max_size = 0;

        for (int i = 0; i < num_neighbors; i++)
        {
            max_size = max_size > A.manager->B2L_rings[i][num_rings_to_send] ? max_size : A.manager->B2L_rings[i][num_rings_to_send];
        }

        IVector matrix_halo_sizes(max_size + 1);

        // Here only using the 1-ring of matrix A
        for (int i = 0; i < num_neighbors; i++)
        {
            // Write the length of the rows in the order of B2L_maps, then calculate row_offsets
            int size = A.manager->B2L_rings[i][num_rings_to_send];

            if (size != 0)
            {
                //matrix_halo_sizes.resize(size+1);
                int num_blocks = min(4096, (size + 127) / 128);
                write_matrix_rowsize_flat<<<num_blocks, 128>>>(A.manager->B2L_maps[i].raw(), P.row_offsets.raw(), size, matrix_halo_sizes.raw());
                thrust_wrapper::exclusive_scan(matrix_halo_sizes.begin(), matrix_halo_sizes.begin() + size + 1, matrix_halo_sizes.begin());
                int nnz_count = matrix_halo_sizes[size];
                // Resize export halo matrix, and copy over the rows
                halo_rows_P_row_offsets[i].resize(size + 1);
                halo_rows_P_col_indices[i].resize(nnz_count);
                halo_rows_P_values[i].resize(nnz_count);
                /* WARNING: Since A is reordered (into interior and boundary nodes), while R and P are not reordered,
                        you must unreorder A when performing R*A*P product in ordre to obtain the correct result. */
                export_matrix_elements_global_flat<32><<<num_blocks, 128>>>(P.row_offsets.raw(), P.values.raw(), P.get_block_size(), A.manager->B2L_maps[i].raw(), matrix_halo_sizes.raw(), halo_rows_P_values[i].raw(), P.col_indices.raw(), halo_rows_P_col_indices[i].raw(), size, RAP.manager->local_to_global_map.raw(), NULL /*A.manager->inverse_renumbering.raw()*/, num_owned_coarse_pts, coarse_base_index);
                thrust::copy(matrix_halo_sizes.begin(), matrix_halo_sizes.begin() + size + 1, halo_rows_P_row_offsets[i].begin());
            }
            else
            {
                halo_rows_P_row_offsets[i].resize(0);
                halo_rows_P_col_indices[i].resize(0);
                halo_rows_P_values[i].resize(0);
            }
        }
    }

    {
        nvtxRange nvtx_exchange_P("exchange_P");
        // Do the exchange with the neighbors
        // On return, halo_rows_P_rows_offsets, halo_rows_P_col_indices and halo_rows_P_values stores the rows of P received from each neighbor (rows needed to perform A*P)
        std::vector<I64Vector> dummy_halo_ids(0);

        // START Exchange matrix halo

        //A.manager->getComms()->exchange_matrix_halo(halo_rows_P_row_offsets, halo_rows_P_col_indices, halo_rows_P_values, dummy_halo_ids, A.manager->neighbors, A.manager->global_id());

        typedef typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig_h::mode)>::Type hmvec_value_type;
        typedef Vector<hmvec_value_type> MVector_h;

        int total = 0;
        int neighbors = A.manager->getComms()->get_neighbors();
        MPI_Comm mpi_comm = A.manager->getComms()->get_mpi_comm();
        MPI_Comm_size(mpi_comm, &total);
        std::vector<IVector_h> local_row_offsets(neighbors);
        std::vector<I64Vector_h> local_col_indices(neighbors);
        std::vector<MVector_h> local_values(neighbors);
        std::vector<IVector_h> send_row_offsets(neighbors);
        std::vector<I64Vector_h> send_col_indices(neighbors);
        std::vector<MVector_h> send_values(neighbors);

        for (int i = 0; i < neighbors; i++)
        {
            send_row_offsets[i] = halo_rows_P_row_offsets[i];
            send_col_indices[i] = halo_rows_P_col_indices[i];
            send_values[i] = halo_rows_P_values[i];
        }

        std::vector<I64Vector_h> local_row_ids(0);
        std::vector<I64Vector_h> send_row_ids(0);

        if (dummy_halo_ids.size() != 0)
        {
            local_row_ids.resize(neighbors);
            send_row_ids.resize(neighbors);

            for (int i = 0; i < neighbors; i++)
            {
                send_row_ids[i] = dummy_halo_ids[i];
            }
        }

        // send metadata
        std::vector<INDEX_TYPE> metadata(neighbors * 2); // num_rows+1, num_nz

        std::vector<MPI_Request> requests(10 * neighbors);

        for (int i = 0; i < 10 * neighbors; i++)
        {
            requests[i] = MPI_REQUEST_NULL;
        }

        for (int i = 0; i < neighbors; i++)
        {
            metadata[i * 2 + 0] = halo_rows_P_row_offsets[i].size();
            metadata[i * 2 + 1] = halo_rows_P_col_indices[i].size();
            MPI_Isend(&metadata[i * 2 + 0], 2, MPI_INT, A.manager->neighbors[i], 0, mpi_comm, &requests[i]);
        }

        // receive metadata
        std::vector<INDEX_TYPE> metadata_recv(2);

        for (int i = 0; i < neighbors; i++)
        {
            MPI_Recv(&metadata_recv[0], 2, MPI_INT, A.manager->neighbors[i], 0, mpi_comm, MPI_STATUSES_IGNORE);
            local_row_offsets[i].resize(metadata_recv[0]);
            local_col_indices[i].resize(metadata_recv[1]);
            local_values[i].resize(metadata_recv[1]);

            if (local_row_ids.size() != 0)
            {
                if (metadata_recv[0] - 1 > 0)
                {
                    local_row_ids[i].resize(metadata_recv[0] - 1); // row_ids is one smaller than row_offsets
                }
            }
        }

        MPI_Waitall(neighbors, &requests[0], MPI_STATUSES_IGNORE); // data is already received, just closing the handles

        // receive matrix data

        for (int i = 0; i < neighbors; i++)
        {
            MPI_Irecv(local_row_offsets[i].raw(), local_row_offsets[i].size(), MPI_INT, A.manager->neighbors[i], 10 * A.manager->neighbors[i] + 0, mpi_comm, &requests[3 * neighbors + i]);
            MPI_Irecv(local_col_indices[i].raw(), local_col_indices[i].size() * sizeof(int64_t), MPI_BYTE, A.manager->neighbors[i], 10 * A.manager->neighbors[i] + 1, mpi_comm, &requests[4 * neighbors + i]);
            MPI_Irecv(local_values[i].raw(), local_values[i].size() * sizeof(double), MPI_BYTE, A.manager->neighbors[i], 10 * A.manager->neighbors[i] + 2, mpi_comm, &requests[5 * neighbors + i]);

            if (send_row_ids.size() != 0)
            {
                MPI_Irecv(local_row_ids[i].raw(), local_row_ids[i].size() * sizeof(int64_t), MPI_BYTE, A.manager->neighbors[i], 10 * A.manager->neighbors[i] + 3, mpi_comm, &requests[7 * neighbors + i]);
            }
        }
        //Note: GPU Direct should use row_offsets[], col_indices[], values[] directly in here:
        // send matrix: row offsets, col indices, values
        for (int i = 0; i < neighbors; i++)
        {
            MPI_Isend(send_row_offsets[i].raw(), send_row_offsets[i].size(), MPI_INT, A.manager->neighbors[i], 10 * A.manager->global_id() + 0, mpi_comm, &requests[i]);
            MPI_Isend(send_col_indices[i].raw(), send_col_indices[i].size() * sizeof(int64_t), MPI_BYTE, A.manager->neighbors[i], 10 * A.manager->global_id() + 1, mpi_comm, &requests[neighbors + i]);
            MPI_Isend(send_values[i].raw(), send_values[i].size() * sizeof(double), MPI_BYTE, A.manager->neighbors[i], 10 * A.manager->global_id() + 2, mpi_comm, &requests[2 * neighbors + i]);

            if (send_row_ids.size() != 0)
            {
                MPI_Isend(send_row_ids[i].raw(), send_row_ids[i].size() * sizeof(int64_t), MPI_BYTE, A.manager->neighbors[i], 10 * A.manager->global_id() + 3, mpi_comm, &requests[6 * neighbors + i]);
            }
        }

        if (dummy_halo_ids.size() != 0)
        {
            MPI_Waitall(8 * neighbors, &requests[0], MPI_STATUSES_IGNORE); //I have to wait for my stuff to be sent too, because I deallocate those matrices upon exditing this function
        }
        else
        {
            MPI_Waitall(6 * neighbors, &requests[0], MPI_STATUSES_IGNORE); //I have to wait for my stuff to be sent too, because I deallocate those matrices upon exditing this function
        }

        //Note: GPU Direct should swap here
        for (int i = 0; i < neighbors; i++)
        {
            halo_rows_P_row_offsets[i] = local_row_offsets[i];
            halo_rows_P_col_indices[i] = local_col_indices[i];
            halo_rows_P_values[i] = local_values[i];

            if (dummy_halo_ids.size() != 0)
            {
                dummy_halo_ids[i] = local_row_ids[i];
            }
        }
    }

    // END Exchange matrix halo

    halo_rows_P_local_col_indices.resize(halo_rows_P_col_indices.size());

    for (int i = 0; i < halo_rows_P_col_indices.size(); i++)
    {
        halo_rows_P_local_col_indices[i].resize(halo_rows_P_col_indices[i].size());
    }

    // Update the list of neighbors "P_neighbors" and the corresponding ranges, offsets
    //update_neighbors_list(&A, &neighbors, &halo_ranges_h, &halo_ranges, &part_offsets_h, &part_offsets, &halo_rows_row_offsets, &halo_rows_col_indices)
    //prep->update_neighbors_list(A, P_neighbors, P_halo_ranges_h, P_halo_ranges, RAP.manager->part_offsets_h, RAP.manager->part_offsets, halo_rows_P_row_offsets, halo_rows_P_col_indices);
    int num_partitions = A.manager->get_num_partitions();
    int my_id = A.manager->global_id();
    int total_halo_rows = 0;
    int total_halo_nnz = 0;
    IVector neighbor_flags(num_partitions, 0);

    for (int i = 0; i < halo_rows_P_row_offsets.size(); i++)
    {
        int num_halo_rows = halo_rows_P_row_offsets[i].size() - 1;

        if (num_halo_rows > 0)
        {
            total_halo_rows += num_halo_rows;
            total_halo_nnz += halo_rows_P_row_offsets[i][num_halo_rows];
            int num_blocks = min(4096, (num_halo_rows + 127) / 128);
            calc_num_neighbors_v2_global_flat<16><<<num_blocks, 128>>>(halo_rows_P_row_offsets[i].raw(), halo_rows_P_col_indices[i].raw(), RAP.manager->part_offsets.raw(), neighbor_flags.raw(), num_partitions, my_id, num_halo_rows);
        }
    }

    IVector_h neighbor_flags_h = neighbor_flags;

    // unset 1-ring neighbors & myself
    for (int i = 0; i < num_neighbors; i++)
    {
        neighbor_flags_h[P_neighbors[i]] = 0;
    }

    neighbor_flags_h[my_id] = 0;
    // this will update neighbor list and halo ranges, note that we don't change 1-ring neighbors order
    //prep->append_neighbors(A, P_neighbors, P_halo_ranges_h, P_halo_ranges, neighbor_flags_h, RAP.manager->part_offsets_h);
    // append_neighbors(&A, &neighbors, &halo_ranges_h, &halo_ranges, &neighbor_flags, &part_offsets_h)

    // This function creates arrays neighbors, halo_ranges_h and halo_ranges
    // base on neighbor_flags
    // Here do an MPI_allgather to figure out which partitions need data from me
    // This is required for non-symmetric matrices
    int num_part = A.manager->get_num_partitions();
    // pack 0/1 array into array of integers (size/32)
    int packed_size = (num_part + 32 - 1) / 32;
    IVector_h packed_nf(packed_size, 0);

    for (int i = 0; i < num_part; i++)
    {
        int packed_pos = i / 32;
        int bit_pos = i % 32;
        packed_nf[packed_pos] += (neighbor_flags[i] << bit_pos);
    }

    // exchange packed neighbor flags
    IVector_h gathered_packed_nf;
    A.manager->getComms()->all_gather_v(packed_nf, gathered_packed_nf, num_part);
    // assign neighbors that have edges to me
    int my_id_pos = my_id / 32;
    int my_id_bit = my_id % 32;

    for (int i = 0; i < num_part; i++)
        if (gathered_packed_nf[i * packed_size + my_id_pos] & (1 << my_id_bit)) // check my bit
        {
            neighbor_flags[i] = 1;
        }

    // compute total number of new neighbors
    int new_neighbors = thrust::reduce(neighbor_flags.begin(), neighbor_flags.end());
    cudaCheckError();
    // save old size
    int old_neighbors = P_neighbors.size();
    P_neighbors.resize(old_neighbors + new_neighbors);
    P_halo_ranges_h.resize(old_neighbors * 2 + new_neighbors * 2);
    // initialize manager->neighbors and manager->halo_ranges_h for the new nodes
    int active_part = old_neighbors;

    // XXX NOT SURE WHAT THIS IS DOING
    //prep->num_part = num_part;

    for (int i = 0; i < num_part; i++)
    {
        if (neighbor_flags[i] > 0)
        {
            P_neighbors[active_part] = i;
            P_halo_ranges_h[2 * active_part] = RAP.manager->part_offsets_h[i];
            P_halo_ranges_h[2 * active_part + 1] = RAP.manager->part_offsets_h[i + 1];
            active_part++;
        }
    }

    P_halo_ranges.resize(old_neighbors * 2 + new_neighbors * 2);
    thrust::copy(P_halo_ranges_h.begin() + old_neighbors * 2, P_halo_ranges_h.end(), P_halo_ranges.begin() + old_neighbors * 2);
    cudaCheckError();

    std::vector<IVector> dummy_boundary_list(0);
    int current_num_rings = 1;
    halo_rows_P_local_col_indices.resize(halo_rows_P_col_indices.size());
    // Convert the global indices in the rows just received to local indices
    // This function updates halo offsets, halo_rows_P_local_col_indices, local_to_global_map
    // This should not modify the existing P manager
    //prep->compute_local_halo_indices( P.row_offsets, P.col_indices, halo_rows_P_row_offsets, halo_rows_P_col_indices, halo_rows_P_local_col_indices, RAP.manager->local_to_global_map,  dummy_boundary_list, P_neighbors, P_halo_ranges_h, P_halo_ranges, P_halo_offsets, coarse_base_index, num_owned_coarse_pts, P.get_num_rows(), current_num_rings);
    //compute_local_halo_indices( &A_row_offsets, &A_col_indices, &halo_row_offsets, &halo_global_indices, &halo_local_indices, &local_to_global, &boundary_lists, &neighbors, &halo_ranges_h, &halo_ranges, &halo_offsets, base_index, index_range, A_num_rows, current_num_rings)

    int base_index = coarse_base_index;
    int index_range = num_owned_coarse_pts;

    // This function checks the halo_col_indices received from the neighbors, and identifies
    // new halo_indices and  updates halo_offsets, local_to_global_map accordingly
    // input: halo row offsets,
    //        halo global column indices
    //        current local to global map
    //        neighbors (new discovered neighbors should already be included)
    //        halo_ranges for the neighbors
    //
    // output: halo offsets,
    //         halo local column indices,
    //         updated local to global map (in place)
    int size = P.get_num_rows();

    //TODO: Are these the optimal block_sizes?
    int num_blocks = min(4096, (size + 127) / 128);

    // compute neighbor offsets & total number of neighbor rows
    int total_rows_of_neighbors = 0;
    std::vector<int> neighbor_offsets_h(num_neighbors + 1, 0);
    int max_neighbor_size = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        total_rows_of_neighbors += P_halo_ranges_h[2 * i + 1] - P_halo_ranges_h[2 * i];
        neighbor_offsets_h[i + 1] = neighbor_offsets_h[i] + P_halo_ranges_h[2 * i + 1] - P_halo_ranges_h[2 * i];
        max_neighbor_size = max_neighbor_size > (P_halo_ranges_h[2 * i + 1] - P_halo_ranges_h[2 * i]) ? max_neighbor_size : (P_halo_ranges_h[2 * i + 1] - P_halo_ranges_h[2 * i]);
    }

    // copy offsets to device
    IVector neighbor_offsets(num_neighbors + 1);
    thrust::copy(neighbor_offsets_h.begin(), neighbor_offsets_h.end(), neighbor_offsets.begin());
    // store flags for all neighbor nodes
    IVector neighbor_nodes(total_rows_of_neighbors);
    // flag neighbor nodes that are in the existing rings as -(local_index)-1
    thrust::fill(neighbor_nodes.begin(), neighbor_nodes.end(), 0);
    cudaCheckError();
    int local_to_global_size = RAP.manager->local_to_global_map.size();
    int num_blocks2 = min(4096, (local_to_global_size + 127) / 128);

    if (local_to_global_size != 0)
    {
        flag_halo_nodes_local_v2_flat<<<num_blocks2, 128>>>(P_halo_ranges.raw(), neighbor_nodes.raw(), neighbor_offsets.raw(), base_index, index_range, num_neighbors, local_to_global_size, RAP.manager->local_to_global_map.raw());
        cudaCheckError();
    }

    // 1) flag NEW neighbors nodes that I need as 1, they will be in the 2nd ring
    // 2) fill out local indices for 1st ring & internal indices
    int num_halos = halo_rows_P_row_offsets.size();

    for (int i = 0; i < num_halos; i++)
    {
        int size = halo_rows_P_col_indices[i].size();

        if (size > 0)
        {
            halo_rows_P_local_col_indices[i].resize(size);
            thrust::fill(halo_rows_P_local_col_indices[i].begin(), halo_rows_P_local_col_indices[i].end(), -1); // fill with -1
            // TODO: launch only on halo rows
            flag_halo_nodes_global_flat<16, 1><<<num_blocks, 128>>>(halo_rows_P_row_offsets[i].raw(), halo_rows_P_row_offsets[i].size() - 1, halo_rows_P_col_indices[i].raw(), P_halo_ranges.raw(), neighbor_nodes.raw(), neighbor_offsets.raw(), base_index, index_range, num_neighbors, halo_rows_P_local_col_indices[i].raw());
        }
    }

    cudaCheckError();
    // replace all negative values with 0 in neighbor flags
    is_less_than_zero pred;
    thrust::replace_if(neighbor_nodes.begin(), neighbor_nodes.end(), pred, 0);
    cudaCheckError();
    // fill halo offsets for the current number of  ring for new neighbors (it will be of size 0)
    int current_num_neighbors = (P_halo_offsets.size() - 1) / current_num_rings;
    ;
    int num_rings = current_num_rings + 1;
    IVector_h new_halo_offsets(num_rings * num_neighbors + 1);

    for (int j = 0; j < current_num_rings; j++)
    {
        for (int i = 0; i <= current_num_neighbors; i++)
        {
            new_halo_offsets[num_neighbors * j + i] = P_halo_offsets[current_num_neighbors * j + i];
        }

        for (int i = current_num_neighbors; i < num_neighbors; i++)
        {
            new_halo_offsets[num_neighbors * j + i + 1] = new_halo_offsets[num_neighbors * j + i];
        }
    }

    P_halo_offsets = new_halo_offsets;
    int ring = current_num_rings;
    int current_num_halo_indices = RAP.manager->local_to_global_map.size();
    //int halo_base = index_range + current_num_halo_indices;
    cudaCheckError();

    // compute neighbors nodes indices (in-place) for each neighbor
    for (int i = 0; i < num_neighbors; i++)
    {
        int last_node, num_halo;

        if (neighbor_offsets_h[i + 1] != neighbor_offsets_h[i])
        {
            last_node = neighbor_nodes[neighbor_offsets_h[i + 1] - 1];
            thrust_wrapper::exclusive_scan(neighbor_nodes.begin() + neighbor_offsets_h[i], neighbor_nodes.begin() + neighbor_offsets_h[i + 1], neighbor_nodes.begin() + neighbor_offsets_h[i]);
            num_halo = neighbor_nodes[neighbor_offsets_h[i + 1] - 1] + last_node;
        }
        else
        {
            num_halo = 0;
            last_node = 0;
        }

        // update halo offsets (L2H)
        P_halo_offsets[ring * num_neighbors + i + 1] = P_halo_offsets[ring * num_neighbors + i] + num_halo;

        // if size = 0 then we don't need to compute it
        if (dummy_boundary_list.size() > 0)
        {
            // create my boundary lists = list of neighbor inner nodes corresponding to halo numbers 0..M
            // basically this will be our 2-ring B2L_maps
            dummy_boundary_list[i].resize(num_halo);
            int size = neighbor_offsets_h[i + 1] - neighbor_offsets_h[i];
            num_blocks = min(4096, (size + 127) / 128);

            if (size > 0)
            {
                populate_B2L_flat<<<num_blocks, 128>>>(neighbor_nodes.raw() + neighbor_offsets_h[i], dummy_boundary_list[i].raw(), last_node, size);
                cudaCheckError();
            }
        }
    }

    cudaCheckError();
    // compute local indices and new local to global mapping
    int new_num_halo_indices = P_halo_offsets[num_rings * num_neighbors] - P_halo_offsets[current_num_rings * num_neighbors];
    RAP.manager->local_to_global_map.resize(current_num_halo_indices + new_num_halo_indices);
    IVector halo_offsets_d(P_halo_offsets.size());
    thrust::copy(P_halo_offsets.begin(), P_halo_offsets.end(), halo_offsets_d.begin());
    cudaCheckError();

    // do this for all ring-1 neighbors
    for (int i = 0; i < num_halos; i++)
    {
        int num_neighbor_rows = halo_rows_P_row_offsets[i].size() - 1;
        num_blocks = min(4096, (num_neighbor_rows + 127) / 128);

        if (num_blocks > 0)
        {
            calc_new_halo_mapping_ring2_flat<16><<<num_blocks, 128>>>(halo_rows_P_row_offsets[i].raw(), num_neighbor_rows, halo_rows_P_col_indices[i].raw(), P_halo_ranges.raw(), base_index, index_range, num_neighbors, halo_offsets_d.raw() + current_num_rings * num_neighbors, neighbor_offsets.raw(), neighbor_nodes.raw(), RAP.manager->local_to_global_map.raw(), halo_rows_P_local_col_indices[i].raw());
        }
    }

    cudaCheckError();

    //prep->append_halo_rows(P, halo_rows_P_row_offsets, halo_rows_P_local_col_indices, halo_rows_P_values);

    // Append the new rows to the matrix P
    P.set_initialized(0);

    int new_num_rows = P.get_num_rows();
    int new_num_nnz = P.row_offsets[P.get_num_rows()];
    int cur_row = P.get_num_rows();
    int cur_offset = new_num_nnz;

    for (int i = 0; i < num_neighbors; i++)
    {
        int size = halo_rows_P_row_offsets[i].size();

        if (size != 0)
        {
            new_num_rows += halo_rows_P_row_offsets[i].size() - 1;
            new_num_nnz += halo_rows_P_local_col_indices[i].size();
        }
    }

    P.resize(new_num_rows, new_num_rows, new_num_nnz, 1, 1, 1);

    for (int i = 0; i < num_neighbors; i++)
    {
        int num_halo_rows = halo_rows_P_row_offsets[i].size() - 1;

        if (num_halo_rows > 0)
        {
            // update halo row offsets in-place
            thrust::transform(halo_rows_P_row_offsets[i].begin(), halo_rows_P_row_offsets[i].end(), thrust::constant_iterator<INDEX_TYPE>(cur_offset), halo_rows_P_row_offsets[i].begin(), thrust::plus<INDEX_TYPE>());
            // insert halo rows
            thrust::copy(halo_rows_P_row_offsets[i].begin(), halo_rows_P_row_offsets[i].end() - 1, P.row_offsets.begin() + cur_row);
            thrust::copy(halo_rows_P_local_col_indices[i].begin(), halo_rows_P_local_col_indices[i].end(), P.col_indices.begin() + cur_offset);
            thrust::copy(halo_rows_P_values[i].begin(), halo_rows_P_values[i].end(), P.values.begin() + cur_offset);
            // update counters
            cur_offset = halo_rows_P_row_offsets[i][num_halo_rows];
            cur_row += num_halo_rows;
        }
    }

    cudaCheckError();
    P.row_offsets[P.get_num_rows()] = cur_offset;
    int num_cols = -1;
    num_cols = thrust_wrapper::reduce(P.col_indices.begin(), P.col_indices.end(), num_cols, thrust::maximum<int>()) + 1;
    cudaCheckError();
    P.set_num_cols(num_cols);

    P.set_initialized(1);

    // At this point, we can compute RAP_full which contains some rows that will need to be sent to neighbors
    // i.e. RAP_full = [ RAP_int ]
    //                 [ RAP_ext ]
    // RAP is [ RAP_int ] + [RAP_ext_received_from_neighbors]
    // We can reuse the serial galerkin product since R, A and P use local indices
    // TODO: latency hiding (i.e. compute RAP_ext, exchange_matrix_halo, then do RAP_int)
    /* WARNING: do not recompute prolongation (P) and restriction (R) when you
                are reusing the level structure (structure_reuse_levels > 0) */
    /* We force for matrix P to have only owned rows to be seen for the correct galerkin product computation*/
    P.set_initialized(0);
    P.set_num_rows(num_owned_fine_pts);
    P.addProps(CSR);
    P.set_initialized(1);

    if (this->isReuseLevel() == false)
    {
        nvtxRange nvtx_transposeR("transpose_R");
        this->R.set_initialized(0);
        this->R.addProps(CSR);
        // Take the tranpose of P to get R
        // Single-GPU transpose, no mpi exchange
        this->R.set_initialized(0);
        P.setView(OWNED);

        transpose(P, this->R, P.get_num_rows());

        if (this->m_min_rows_latency_hiding < 0 || this->R.get_num_rows() < this->m_min_rows_latency_hiding)
        {
            // This will cause bsrmv_with_mask_restriction to not do latency hiding
            this->R.setInteriorView(OWNED);
            this->R.setExteriorView(OWNED);
        }

        this->R.set_initialized(1);
        this->R.set_initialized(1);
    }

    Matrix<TConfig_d> RAP_full;
    // Initialize the workspace needed for galerkin product
    void *wk = AMG_Level<TConfig_d>::amg->getCsrWorkspace();

    if (wk == NULL)
    {
        wk = CSR_Multiply<TConfig_d>::csr_workspace_create(*(AMG_Level<TConfig_d>::amg->m_cfg), AMG_Level<TConfig_d>::amg->m_cfg_scope);
        AMG_Level<TConfig_d>::amg->setCsrWorkspace(wk);
    }

    {
        nvtxRange nvtx_galerkin("csr_galerkin_product");
        // Single-GPU RAP, no mpi exchange
        RAP_full.set_initialized(0);
        /* WARNING: Since A is reordered (into interior and boundary nodes), while R and P are not reordered,
                    you must unreorder A when performing R*A*P product in ordre to obtain the correct result. */
        CSR_Multiply<TConfig_d>::csr_galerkin_product(
            this->R,
            this->getA(),
            this->P,
            RAP_full,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            wk);
    }
    RAP_full.set_initialized(1);
    this->Profile.toc("computeA");
    {
        nvtxRange nvtx_exchangeRAP("exchange_RAP");
        // ----------------------------------------------------------------------------------------------
        // Now, send rows of RAP_full requireq by neighbors, received rows from neighbors and create RAP
        // ----------------------------------------------------------------------------------------------
        prep->exchange_RAP_ext(RAP, RAP_full, A, this->P, P_halo_offsets, RAP.manager->local_to_global_map, P_neighbors, P_halo_ranges_h, P_halo_ranges, RAP.manager->part_offsets_h, RAP.manager->part_offsets, num_owned_coarse_pts, RAP.manager->part_offsets_h[my_rank], wk);
    }
    // Delete temporary distributed arranger
    delete prep;

    /* WARNING: The RAP matrix generated at this point contains extra rows (that correspond to rows of R,
       that was obtained by locally transposing P). This rows are ignored by setting the # of matrix rows
       to be smaller, so that they correspond to number of owned coarse nodes. This should be fine, but
       it leaves holes in the matrix as there might be columns that belong to the extra rows that now do not
       belong to the smaller matrix with number of owned coarse nodes rows. The same is trued about the
       local_to_global_map. These two data structures match at this point. However, in the next calls
       local_to_global (exclusively) will be used to geberate B2L_maps (wihtout going through column indices)
       which creates extra elements in the B2L that simply do not exist in the new matrices. I strongly suspect
       this is the reason fore the bug. The below fix simply compresses the matrix so that there are no holes
       in it, or in the local_2_global_map. */
    //mark local_to_global_columns that exist in the owned coarse nodes rows.
    IndexType nrow = RAP.get_num_rows();
    IndexType ncol = RAP.get_num_cols();
    IndexType nl2g = ncol - nrow;

    if (nl2g > 0)
    {
        nvtxRange nvtx_deleterows("delete_rows");
        IVector l2g_p(nl2g + 1, 0); //+1 is needed for prefix_sum/exclusive_scan
        I64Vector l2g_t(nl2g, 0);
        IndexType nblocks = (nrow + AMGX_CAL_BLOCK_SIZE - 1) / AMGX_CAL_BLOCK_SIZE;

        if (nblocks > 0)
        {
            flag_existing_local_to_global_columns<int><<<nblocks, AMGX_CAL_BLOCK_SIZE>>>(nrow, RAP.row_offsets.raw(), RAP.col_indices.raw(), l2g_p.raw());
        }

        //create a pointer map for their location using prefix sum
        thrust_wrapper::exclusive_scan(l2g_p.begin(), l2g_p.end(), l2g_p.begin());
        int new_nl2g = l2g_p[nl2g];

        //compress the columns using the pointer map
        if (nblocks > 0)
            compress_existing_local_columns<int><<<nblocks, AMGX_CAL_BLOCK_SIZE>>>(nrow, RAP.row_offsets.raw(), RAP.col_indices.raw(), l2g_p.raw());

        //adjust matrix size (number of columns) accordingly
        RAP.set_initialized(0);
        RAP.set_num_cols(nrow + new_nl2g);
        RAP.set_initialized(1);
        //compress local_to_global_map using the pointer map
        nblocks = (nl2g + AMGX_CAL_BLOCK_SIZE - 1) / AMGX_CAL_BLOCK_SIZE;

        if (nblocks > 0)
        {
            compress_existing_local_to_global_columns<int, int64_t><<<nblocks, AMGX_CAL_BLOCK_SIZE>>>(nl2g, RAP.manager->local_to_global_map.raw(), l2g_t.raw(), l2g_p.raw());
        }

        thrust::copy(l2g_t.begin(), l2g_t.begin() + new_nl2g, RAP.manager->local_to_global_map.begin());

        //adjust local_to_global_map size accordingly
        RAP.manager->local_to_global_map.resize(new_nl2g);
    }

    // we also need to renumber columns of P and rows or R correspondingly since we changed RAP halo columns
    // for R we just keep track of renumbering in and exchange proper vectors in restriction
    // for P we actually need to renumber columns for prolongation:
    if (A.is_matrix_distributed() && this->A->manager->get_num_partitions() > 1)
    {
        RAP.set_initialized(0);
        {
            nvtxRange nvtx_renum("renumber_RAP");
            // Renumber the owned nodes as interior and boundary (renumber rows and columns)
            // We are passing reuse flag to not create neighbours list from scratch, but rather update based on new halos
            RAP.manager->renumberMatrixOneRing(this->isReuseLevel());
            // Renumber the column indices of P and shuffle rows of P
            RAP.manager->renumber_P_R(this->P, this->R, A);
            // Create the B2L_maps for RAP
        }
        {
            nvtxRange fdafds("createOneRingHaloRows RAP");

            DistributedArranger<TConfig_d> *prep = new DistributedArranger<TConfig_d>;
            //prep->create_one_ring_halo_rows(RAP);

            // what it does:
            //   appends halo rows to matrix A
            //   creates new B2L_rings, B2L_maps and L2H_maps
            // input:
            //   matrix A and 1-ring B2L_maps
            int num_partitions = RAP.manager->get_num_partitions();
            int my_id = RAP.manager->global_id();
            int num_neighbors = RAP.manager->B2L_maps.size();
            int num_ring1_neighbors = num_neighbors;
            int num_ring1_indices = RAP.manager->local_to_global_map.size();
            std::vector<IVector> halo_row_offsets(num_neighbors);
            std::vector<I64Vector> halo_global_indices(num_neighbors);
            std::vector<IVector> halo_local_indices(num_neighbors);
            std::vector<MVector> halo_values(num_neighbors);
            // step 1: setup halo rows with global indices
            // in this function we assume that we only have 1-ring so far
            int neighbors = RAP.manager->num_neighbors();

            for (int i = 0; i < neighbors; i++)
            {
                // compute row offsets and nnz count
                int size = RAP.manager->B2L_maps[i].size();
                halo_row_offsets[i].resize(size + 1);

                if (size > 0)
                {
                    int num_blocks = min(4096, (size + 127) / 128);
                    write_matrix_rowsize_flat<<<num_blocks, 128>>>(RAP.manager->B2L_maps[i].raw(), RAP.row_offsets.raw(), size, halo_row_offsets[i].raw());
                    thrust_wrapper::exclusive_scan(halo_row_offsets[i].begin(), halo_row_offsets[i].begin() + size + 1, halo_row_offsets[i].begin());
                    // compute global indices
                    int nnz_count = halo_row_offsets[i][size];
                    halo_global_indices[i].resize(nnz_count);
                    halo_values[i].resize(nnz_count);
                    export_matrix_elements_global_flat<32><<<num_blocks, 128>>>(RAP.row_offsets.raw(), RAP.values.raw(), RAP.get_block_size(), RAP.manager->B2L_maps[i].raw(), halo_row_offsets[i].raw(), halo_values[i].raw(), RAP.col_indices.raw(), halo_global_indices[i].raw(), size, RAP.manager->local_to_global_map.raw(), NULL, RAP.get_num_rows(), RAP.manager->base_index());
                }
            }
            // step 2: exchange halo rows
            std::vector<I64Vector> dummy_halo_ids(0);
            RAP.manager->getComms()->exchange_matrix_halo(halo_row_offsets, halo_global_indices, halo_values, dummy_halo_ids, RAP.manager->neighbors, RAP.manager->global_id());
            // step 3: append the list  f neighbors with the new ring-2 neighbors

            //update_neighbors_list(RAP, RAP.manager->neighbors, RAP.manager->halo_ranges_h, RAP.manager->halo_ranges, RAP.manager->part_offsets_h, RAP.manager->part_offsets, halo_row_offsets, halo_global_indices);
            int total_halo_rows = 0;
            int total_halo_nnz = 0;
            IVector neighbor_flags(num_partitions, 0);

            for (int i = 0; i < halo_row_offsets.size(); i++)
            {
                int num_halo_rows = halo_row_offsets[i].size() - 1;

                if (num_halo_rows > 0)
                {
                    total_halo_rows += num_halo_rows;
                    total_halo_nnz += halo_row_offsets[i][num_halo_rows];
                    int num_blocks = min(4096, (num_halo_rows + 127) / 128);
                    calc_num_neighbors_v2_global_flat<16><<<num_blocks, 128>>>(halo_row_offsets[i].raw(), halo_global_indices[i].raw(),
                                                                               RAP.manager->part_offsets.raw(), neighbor_flags.raw(), num_partitions, my_id, num_halo_rows);
                }
            }

            cudaCheckError();
            IVector_h neighbor_flags_h = neighbor_flags;

            // unset 1-ring neighbors & myself
            for (int i = 0; i < num_neighbors; i++)
            {
                neighbor_flags_h[RAP.manager->neighbors[i]] = 0;
            }

            neighbor_flags_h[my_id] = 0;

            // this will update neighbor list and halo ranges, note that we don't change 1-ring neighbors order
            //append_neighbors(RAP, neighbors, RAP.manager->halo_ranges_h, RAP.manager->halo_ranges, neighbor_flags_h, RAP.manager->part_offsets_h);

            // This function creates arrays neighbors, halo_ranges_h and halo_ranges
            // base on neighbor_flags
            // Here do an MPI_allgather to figure out which partitions need data from me
            // This is required for non-symmetric matrices
            int num_part = RAP.manager->get_num_partitions();
            // pack 0/1 array into array of integers (size/32)
            int packed_size = (num_part + 32 - 1) / 32;
            IVector_h packed_nf(packed_size, 0);

            for (int i = 0; i < num_part; i++)
            {
                int packed_pos = i / 32;
                int bit_pos = i % 32;
                packed_nf[packed_pos] += (neighbor_flags[i] << bit_pos);
            }

            // exchange packed neighbor flags
            IVector_h gathered_packed_nf;
            RAP.manager->getComms()->all_gather_v(packed_nf, gathered_packed_nf, num_part);
            // assign neighbors that have edges to me
            int my_id_pos = my_id / 32;
            int my_id_bit = my_id % 32;

            for (int i = 0; i < num_part; i++)
                if (gathered_packed_nf[i * packed_size + my_id_pos] & (1 << my_id_bit)) // check my bit
                {
                    neighbor_flags[i] = 1;
                }

            // compute total number of new neighbors
            int new_neighbors = thrust::reduce(neighbor_flags.begin(), neighbor_flags.end());
            cudaCheckError();
            // save old size
            int old_neighbors = RAP.manager->neighbors.size();
            RAP.manager->neighbors.resize(old_neighbors + new_neighbors);
            RAP.manager->halo_ranges_h.resize(old_neighbors * 2 + new_neighbors * 2);
            // initialize manager->neighbors and manager->halo_ranges_h for the new nodes
            int active_part = old_neighbors;

            //this->num_part = num_part;

            for (int i = 0; i < num_part; i++)
            {
                if (neighbor_flags[i] > 0)
                {
                    RAP.manager->neighbors[active_part] = i;
                    RAP.manager->halo_ranges_h[2 * active_part] = RAP.manager->part_offsets_h[i];
                    RAP.manager->halo_ranges_h[2 * active_part + 1] = RAP.manager->part_offsets_h[i + 1];
                    active_part++;
                }
            }

            RAP.manager->halo_ranges.resize(old_neighbors * 2 + new_neighbors * 2);
            thrust::copy(RAP.manager->halo_ranges_h.begin() + old_neighbors * 2, RAP.manager->halo_ranges_h.end(), RAP.manager->halo_ranges.begin() + old_neighbors * 2);
            cudaCheckError();

            num_neighbors = RAP.manager->neighbors.size();
            // step 4: mark neighbor nodes, create local halo indices for the 2nd ring
            // this function also creates boundary lists for the 2nd ring if boundary lists array size > 0
            std::vector<IVector> boundary_lists(num_neighbors);

            //compute_local_halo_indices(RAP.row_offsets, RAP.col_indices, halo_row_offsets, halo_global_indices, halo_local_indices, RAP.manager->local_to_global_map, boundary_lists,  RAP.manager->neighbors, RAP.manager->halo_ranges_h, RAP.manager->halo_ranges, RAP.manager->halo_offsets, RAP.manager->base_index(), RAP.manager->index_range(), RAP.get_num_rows(), 1);

            // This function checks the halo_col_indices received from the neighbors, and identifies
            // new halo_indices and  updates halo_offsets, local_to_global_map accordingly
            // input: halo row offsets,
            //        halo global column indices
            //        current local to global map
            //        neighbors (new discovered neighbors should already be included)
            //        halo_ranges for the neighbors
            //
            // output: halo offsets,
            //         halo local column indices,
            //         updated local to global map (in place)
            int size = RAP.get_num_rows();
            //TODO: Are these the optimal block_sizes?
            int num_blocks = min(4096, (size + 127) / 128);
            // compute neighbor offsets & total number of neighbor rows
            int total_rows_of_neighbors = 0;
            std::vector<int> neighbor_offsets_h(num_neighbors + 1, 0);
            int max_neighbor_size = 0;

            for (int i = 0; i < num_neighbors; i++)
            {
                total_rows_of_neighbors += RAP.manager->halo_ranges_h[2 * i + 1] - RAP.manager->halo_ranges_h[2 * i];
                neighbor_offsets_h[i + 1] = neighbor_offsets_h[i] + RAP.manager->halo_ranges_h[2 * i + 1] - RAP.manager->halo_ranges_h[2 * i];
                max_neighbor_size = max_neighbor_size > (RAP.manager->halo_ranges_h[2 * i + 1] - RAP.manager->halo_ranges_h[2 * i]) ? max_neighbor_size : (RAP.manager->halo_ranges_h[2 * i + 1] - RAP.manager->halo_ranges_h[2 * i]);
            }

            // copy offsets to device
            IVector neighbor_offsets(num_neighbors + 1);
            thrust::copy(neighbor_offsets_h.begin(), neighbor_offsets_h.end(), neighbor_offsets.begin());
            // store flags for all neighbor nodes
            IVector neighbor_nodes(total_rows_of_neighbors);
            // flag neighbor nodes that are in the existing rings as -(local_index)-1
            thrust::fill(neighbor_nodes.begin(), neighbor_nodes.end(), 0);
            cudaCheckError();
            int local_to_global_size = RAP.manager->local_to_global_map.size();
            int num_blocks2 = min(4096, (local_to_global_size + 127) / 128);

            if (local_to_global_size != 0)
            {
                flag_halo_nodes_local_v2_flat<<<num_blocks2, 128>>>(RAP.manager->halo_ranges.raw(), neighbor_nodes.raw(), neighbor_offsets.raw(), base_index, index_range, num_neighbors, local_to_global_size, RAP.manager->local_to_global_map.raw());
                cudaCheckError();
            }

            // 1) flag NEW neighbors nodes that I need as 1, they will be in the 2nd ring
            // 2) fill out local indices for 1st ring & internal indices
            int num_halos = halo_row_offsets.size();

            for (int i = 0; i < num_halos; i++)
            {
                int size = halo_global_indices[i].size();

                if (size > 0)
                {
                    halo_local_indices[i].resize(size);
                    thrust::fill(halo_local_indices[i].begin(), halo_local_indices[i].end(), -1); // fill with -1
                    // TODO: launch only on halo rows
                    flag_halo_nodes_global_flat<16, 1><<<num_blocks, 128>>>(halo_row_offsets[i].raw(), halo_row_offsets[i].size() - 1, halo_global_indices[i].raw(), RAP.manager->halo_ranges.raw(), neighbor_nodes.raw(), neighbor_offsets.raw(), base_index, index_range, num_neighbors, halo_local_indices[i].raw());
                }
            }

            cudaCheckError();
            // replace all negative values with 0 in neighbor flags
            is_less_than_zero pred;
            thrust::replace_if(neighbor_nodes.begin(), neighbor_nodes.end(), pred, 0);
            cudaCheckError();
            // fill halo offsets for the current number of  ring for new neighbors (it will be of size 0)
            int current_num_neighbors = (RAP.manager->halo_offsets.size() - 1) / current_num_rings;
            ;
            int num_rings = current_num_rings + 1;
            IVector_h new_halo_offsets(num_rings * num_neighbors + 1);

            for (int j = 0; j < current_num_rings; j++)
            {
                for (int i = 0; i <= current_num_neighbors; i++)
                {
                    new_halo_offsets[num_neighbors * j + i] = RAP.manager->halo_offsets[current_num_neighbors * j + i];
                }

                for (int i = current_num_neighbors; i < num_neighbors; i++)
                {
                    new_halo_offsets[num_neighbors * j + i + 1] = new_halo_offsets[num_neighbors * j + i];
                }
            }

            RAP.manager->halo_offsets = new_halo_offsets;
            int ring = current_num_rings;
            int current_num_halo_indices = RAP.manager->local_to_global_map.size();
            //int halo_base = index_range + current_num_halo_indices;
            cudaCheckError();

            // compute neighbors nodes indices (in-place) for each neighbor
            for (int i = 0; i < num_neighbors; i++)
            {
                int last_node, num_halo;

                if (neighbor_offsets_h[i + 1] != neighbor_offsets_h[i])
                {
                    last_node = neighbor_nodes[neighbor_offsets_h[i + 1] - 1];
                    thrust_wrapper::exclusive_scan(neighbor_nodes.begin() + neighbor_offsets_h[i], neighbor_nodes.begin() + neighbor_offsets_h[i + 1], neighbor_nodes.begin() + neighbor_offsets_h[i]);
                    num_halo = neighbor_nodes[neighbor_offsets_h[i + 1] - 1] + last_node;
                }
                else
                {
                    num_halo = 0;
                    last_node = 0;
                }

                // update halo offsets (L2H)
                RAP.manager->halo_offsets[ring * num_neighbors + i + 1] = RAP.manager->halo_offsets[ring * num_neighbors + i] + num_halo;

                // if size = 0 then we don't need to compute it
                if (boundary_lists.size() > 0)
                {
                    // create my boundary lists = list of neighbor inner nodes corresponding to halo numbers 0..M
                    // basically this will be our 2-ring B2L_maps
                    boundary_lists[i].resize(num_halo);
                    int size = neighbor_offsets_h[i + 1] - neighbor_offsets_h[i];
                    num_blocks = min(4096, (size + 127) / 128);

                    if (size > 0)
                    {
                        populate_B2L_flat<<<num_blocks, 128>>>(neighbor_nodes.raw() + neighbor_offsets_h[i], boundary_lists[i].raw(), last_node, size);
                        cudaCheckError();
                    }
                }
            }

            cudaCheckError();
            // compute local indices and new local to global mapping
            int new_num_halo_indices = RAP.manager->halo_offsets[num_rings * num_neighbors] - RAP.manager->halo_offsets[current_num_rings * num_neighbors];
            RAP.manager->local_to_global_map.resize(current_num_halo_indices + new_num_halo_indices);
            IVector halo_offsets_d(RAP.manager->halo_offsets.size());
            thrust::copy(RAP.manager->halo_offsets.begin(), RAP.manager->halo_offsets.end(), halo_offsets_d.begin());
            cudaCheckError();

            // do this for all ring-1 neighbors
            for (int i = 0; i < num_halos; i++)
            {
                int num_neighbor_rows = halo_row_offsets[i].size() - 1;
                num_blocks = min(4096, (num_neighbor_rows + 127) / 128);

                if (num_blocks > 0)
                {
                    calc_new_halo_mapping_ring2_flat<16><<<num_blocks, 128>>>(
                        halo_row_offsets[i].raw(), num_neighbor_rows, halo_global_indices[i].raw(),                             // halo rows to process
                        RAP.manager->halo_ranges.raw(), base_index, index_range, num_neighbors,                                 // ranges and # of neighbors
                        halo_offsets_d.raw() + current_num_rings * num_neighbors, neighbor_offsets.raw(), neighbor_nodes.raw(), // halo offsets, neighbor offsets and indices
                        RAP.manager->local_to_global_map.raw(), halo_local_indices[i].raw());                                   // output
                }
            }

            cudaCheckError();

            // update renumbering arrays (set identity for 2nd ring)
            // step 5: update L2H maps = identity
            RAP.manager->getComms()->set_neighbors(RAP.manager->neighbors.size());
            // step 6: receive neighbors boundary lists for 2-ring b2l maps
            RAP.manager->getComms()->exchange_vectors(boundary_lists, RAP, 0);
            // step 7: update B2L rings & B2L maps
            int rings = 2;
            RAP.manager->B2L_rings.resize(num_neighbors);
            RAP.manager->B2L_maps.resize(num_neighbors);
            // modify 1-ring where necessary
            ring = 0;

            for (int i = 0; i < num_neighbors; i++)
            {
                // set size to 0 for 2-ring neighbors only
                if (i >= num_ring1_neighbors)
                {
                    RAP.manager->B2L_maps[i].resize(0);
                }

                RAP.manager->B2L_rings[i].resize(rings + 1);
                RAP.manager->B2L_rings[i][0] = 0;
                RAP.manager->B2L_rings[i][ring + 1] = RAP.manager->B2L_maps[i].size();
            }

            // fill up 2-ring maps
            ring = 1;

            for (int i = 0; i < num_neighbors; i++)
            {
                // append 2nd ring
                int ring1_size = RAP.manager->B2L_maps[i].size();
                RAP.manager->B2L_maps[i].resize(ring1_size + boundary_lists[i].size());
                thrust::copy(boundary_lists[i].begin(), boundary_lists[i].end(), RAP.manager->B2L_maps[i].begin() + ring1_size);
                RAP.manager->B2L_rings[i][ring + 1] = RAP.manager->B2L_maps[i].size();
            }

            cudaCheckError();
            // Compute the total number of bdy rows in each ring
            std::vector<IVector_h> B2L_maps_offsets_h(2);
            RAP.manager->B2L_rings_sizes.resize(2);

            for (int k = 0; k < 2; k++)
            {
                B2L_maps_offsets_h[k].resize(num_neighbors + 1);
                B2L_maps_offsets_h[k][0] = 0;

                for (int j = 0; j < num_neighbors; j++)
                {
                    B2L_maps_offsets_h[k][j + 1] = B2L_maps_offsets_h[k][j] + RAP.manager->B2L_rings[j][k + 1];
                }

                RAP.manager->B2L_rings_sizes[k] = B2L_maps_offsets_h[k][num_neighbors];
            }

            // Copy maps_offsets to device
            RAP.manager->B2L_maps_offsets.resize(2);

            for (int i = 0; i < 2; i++)
            {
                RAP.manager->B2L_maps_offsets[i] = B2L_maps_offsets_h[i];
            }

            // Store the B2L_maps ptrs on the device
            std::vector<int *> B2L_maps_ptrs_h(num_neighbors);

            for (int j = 0; j < num_neighbors; j++)
            {
                B2L_maps_ptrs_h[j] = RAP.manager->B2L_maps[j].raw();
            }

            RAP.manager->B2L_maps_ptrs = B2L_maps_ptrs_h;
            // step 8: append halo_rows to matrix RAP
            // compute new # of rows & nnz
            //prep->append_halo_rows(RAP, halo_row_offsets, halo_local_indices, halo_values);
            int new_num_rows = RAP.get_num_rows();
            int new_num_nnz = RAP.row_offsets[RAP.get_num_rows()];
            int cur_row = RAP.get_num_rows();
            int cur_offset = new_num_nnz;

            for (int i = 0; i < num_neighbors; i++)
            {
                int size = halo_row_offsets[i].size();

                if (size != 0)
                {
                    new_num_rows += halo_row_offsets[i].size() - 1;
                    new_num_nnz += halo_local_indices[i].size();
                }
            }

            RAP.resize(new_num_rows, new_num_rows, new_num_nnz, 1, 1, 1);

            for (int i = 0; i < num_neighbors; i++)
            {
                int num_halo_rows = halo_row_offsets[i].size() - 1;

                if (num_halo_rows > 0)
                {
                    // update halo row offsets in-place
                    thrust::transform(halo_row_offsets[i].begin(), halo_row_offsets[i].end(), thrust::constant_iterator<INDEX_TYPE>(cur_offset), halo_row_offsets[i].begin(), thrust::plus<INDEX_TYPE>());
                    // insert halo rows
                    thrust::copy(halo_row_offsets[i].begin(), halo_row_offsets[i].end() - 1, RAP.row_offsets.begin() + cur_row);
                    thrust::copy(halo_local_indices[i].begin(), halo_local_indices[i].end(), RAP.col_indices.begin() + cur_offset);
                    thrust::copy(halo_values[i].begin(), halo_values[i].end(), RAP.values.begin() + cur_offset);
                    // update counters
                    cur_offset = halo_row_offsets[i][num_halo_rows];
                    cur_row += num_halo_rows;
                }
            }

            cudaCheckError();
            RAP.row_offsets[RAP.get_num_rows()] = cur_offset;
            int num_cols = -1;
            num_cols = thrust_wrapper::reduce(RAP.col_indices.begin(), RAP.col_indices.end(), num_cols, thrust::maximum<int>()) + 1;
            cudaCheckError();
            RAP.set_num_cols(num_cols);
            // initialize the manager
            RAP.manager->set_initialized(RAP.row_offsets);
            // Compute the diagonal
            // TODO: Should only compute diagonal of 1-ring halo rows
            ViewType oldView = RAP.currentView();
            RAP.setView(FULL);
            RAP.set_allow_recompute_diag(true);
            RAP.computeDiagonal();
            RAP.setView(oldView);
            // the following steps are necessary only for latency hiding/renumbering, i.e. to use reorder_matrix()
            delete prep;
        }
        RAP.manager->getComms()->set_neighbors(RAP.manager->num_neighbors());
        RAP.setView(OWNED);
        RAP.set_initialized(1);

        // update # of columns in P - this is necessary for correct CSR multiply
        this->P.set_initialized(0);
        int new_num_cols = thrust_wrapper::reduce(this->P.col_indices.begin(), this->P.col_indices.end(), int(0), thrust::maximum<int>()) + 1;
        cudaCheckError();
        this->P.set_num_cols(new_num_cols);
        this->P.set_initialized(1);
    }

    RAP.copyAuxData(&A);

    if (!A.is_matrix_singleGPU() && RAP.manager == NULL)
    {
        RAP.manager = new DistributedManager<TConfig_d>();
    }

    if (this->getA().is_matrix_singleGPU())
    {
        this->m_next_level_size = this->getNextLevel(typename Matrix<TConfig_d>::memory_space())->getA().get_num_rows() * this->getNextLevel(typename Matrix<TConfig_d>::memory_space())->getA().get_block_dimy();
    }
    else
    {
        // m_next_level_size is the size that will be used to allocate xc, bc vectors
        int size, offset;
        this->getNextLevel(typename Matrix<TConfig_d>::memory_space())->getA().getOffsetAndSizeForView(FULL, &offset, &size);
        this->m_next_level_size = size * this->getNextLevel(typename Matrix<TConfig_d>::memory_space())->getA().get_block_dimy();
    }

    printf("FLATTERNED COMPLETE\n");
}

/****************************************
 * Explict instantiations
 ***************************************/
#define AMGX_CASE_LINE(CASE) template class Classical_AMG_Level_Base<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class Classical_AMG_Level<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace classical

} // namespace amgx
