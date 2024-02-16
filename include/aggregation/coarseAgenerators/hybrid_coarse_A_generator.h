// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <aggregation/coarseAgenerators/coarse_A_generator.h>
#include <matrix.h>

namespace amgx
{
namespace aggregation
{

template <class T_Config> class HybridCoarseAGenerator;

template <class T_Config>
class HybridCoarseAGeneratorBase: public CoarseAGenerator<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef typename T_Config::MatPrec ValueType;
        typedef typename T_Config::IndPrec IndexType;
        typedef typename Matrix<T_Config>::IVector IVector;

        HybridCoarseAGeneratorBase();

        void computeAOperator(const Matrix<T_Config> &A, Matrix<T_Config> &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates);
    protected:
        virtual void computeAOperator_4x4(const Matrix<T_Config> &A, Matrix<T_Config> &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates) = 0;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class HybridCoarseAGenerator<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public HybridCoarseAGeneratorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Matrix_d;
        typedef typename Matrix_h::value_type ValueType;
        typedef typename Matrix_h::index_type IndexType;
        typedef typename Matrix_h::IVector IVector;
        HybridCoarseAGenerator() : HybridCoarseAGeneratorBase<TConfig>() {}
    private:
        void computeAOperator_4x4(const Matrix_h &A, Matrix_h &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates);
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class HybridCoarseAGenerator<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >: public HybridCoarseAGeneratorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef Matrix< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Matrix_h;
        typedef Matrix< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Matrix_d;
        typedef typename Matrix_d::value_type ValueType;
        typedef typename Matrix_d::index_type IndexType;
        typedef typename Matrix_d::IVector IVector;
        HybridCoarseAGenerator() : HybridCoarseAGeneratorBase<TConfig>() {}
    private:
        void computeAOperator_4x4(const Matrix_d &A, Matrix_d &Ac, const IVector &aggregates, const IVector &R_row_offsets, const IVector &R_column_indices, const int num_aggregates);
};

template<class T_Config>
class HybridCoarseAGeneratorFactory : public CoarseAGeneratorFactory<T_Config>
{
    public:
        CoarseAGenerator<T_Config> *create(AMG_Config &, const std::string &) { return new HybridCoarseAGenerator<T_Config>; }
};
}
}
