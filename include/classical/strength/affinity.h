// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <classical/strength/strength_base.h>
//#include <solvers/gauss_seidel_solver.h>
#include <solvers/multicolor_gauss_seidel_solver.h>

namespace amgx
{

using namespace multicolor_gauss_seidel_solver;

template <class T_Config> class  Strength_Affinity;

template <class T_Config>
class Strength_AffinityBase : public Strength_BaseBase<T_Config>
{
        typedef T_Config TConfig;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename T_Config::VecPrec ValueTypeB;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<TConfig>::MVector MVector;
    public:
        Strength_AffinityBase(AMG_Config &cfg, const std::string &cfg_scope) : Strength_BaseBase<T_Config>(cfg, cfg_scope) {};
        __host__ __device__
        bool strongly_connected(ValueType val, ValueType threshold, ValueType diagonal)
        {
            return (diagonal < 0.0) ?  val > threshold : val < threshold;
        }
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class  Strength_Affinity< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public  Strength_AffinityBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef typename TConfig_h::MatPrec ValueType;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<TConfig_h>::MVector VVector;
        typedef Matrix<TConfig_h> Matrix_h;
    public:
        Strength_Affinity(AMG_Config &cfg, const std::string &cfg_scope) : Strength_AffinityBase<TConfig_h>(cfg, cfg_scope) {}
    private:
        virtual void computeStrongConnectionsAndWeights_1x1(Matrix_h &A,
                BVector &s_con,
                FVector &weights,
                const double max_row_sum)
        {
            FatalError("Strength affinity: computeStrongConnectionsAndWeights_1x1 not implemented on CPU, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
        }
        virtual void computeWeights_1x1(Matrix_h &S,
                                        FVector &weights)
        {
            FatalError("Strength affinity: computeWeights_1x1 not implemented on CPU, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
        }
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class  Strength_Affinity< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public  Strength_AffinityBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_device, AMGX_vecFloat, t_matPrec, t_indPrec> TConfig_df;
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type> BVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<TConfig_d>::MVector VVector;
        typedef Matrix<TConfig_d> Matrix_d;
    public:
        int n_TV;
        int affinity_iters;
        // setup mixed precision solver
        MulticolorGaussSeidelSolver<TConfig_d> *solver;
        Strength_Affinity(AMG_Config &cfg, const std::string &cfg_scope);
        ~Strength_Affinity() {delete solver;};
    private:
        Matrix<TConfig_d> m_aff;
        Vector<TConfig_d> m_x;
        Vector<TConfig_d> m_rhs;
        VVector m_aff_values;

        void computeStrongConnectionsAndWeights_1x1(Matrix_d &A,
                BVector &s_con,
                FVector &weights,
                const double max_row_sum);
        void computeWeights_1x1(Matrix_d &S,
                                FVector &weights)
        {
            FatalError("Strength affinity: computeWeights_1x1 not implemented on GPU, exiting", AMGX_ERR_NOT_SUPPORTED_TARGET);
        }
};

template<class T_Config>
class Strength_Affinity_StrengthFactory: public StrengthFactory<T_Config>
{
    public:
        Strength<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new Strength_Affinity<T_Config>(cfg, cfg_scope); }
};

} // namespace amgx