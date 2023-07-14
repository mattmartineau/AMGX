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

#pragma once

#include <classical/selectors/selector.h>

namespace amgx
{

namespace classical
{

template <class T_Config> class  CR_Selector;

template <class T_Config>
class CR_SelectorBase : public Selector<T_Config>
{
        typedef T_Config TConfig;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecBool>::Type>  BVector;
        typedef Vector<typename TConfig::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<T_Config>::MVector VVector;
        typedef typename Matrix<T_Config>::IVector IVector;


    public:
        void markCoarseFinePoints(Matrix<T_Config> &A,
                                  FVector &weights,
                                  const BVector &s_con,
                                  IVector &cf_map,
                                  IVector &scratch,
                                  int cf_map_init = 0);

        CR_SelectorBase(AMG_Config &cfg, const std::string &cfg_scope) : 
          Selector<T_Config>(cfg, cfg_scope) {}

        virtual ~CR_SelectorBase() {};

    protected:
        virtual void markCoarseFinePoints_1x1(Matrix<T_Config> &A,
                                              IVector &cf_map) = 0;
};


// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class CR_Selector< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > :
    public CR_SelectorBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef typename TConfig_h::MatPrec ValueType;
        typedef typename TConfig_h::IndPrec IndexType;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecBool>::Type>  BVector;
        typedef Vector<typename TConfig_h::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<TConfig_h>::MVector VVector;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename Matrix_h::IVector IVector;

    public:
        CR_Selector(AMG_Config &cfg, const std::string &cfg_scope) : 
          CR_SelectorBase<TConfig_h>(cfg, cfg_scope) {}
    private:
        void markCoarseFinePoints_1x1(Matrix_h &A,
                                      IVector &cf_map)
        {
            FatalError("CR selector is not implemented on host", AMGX_ERR_NOT_IMPLEMENTED);
        }
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class CR_Selector< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > :
    public CR_SelectorBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef typename TConfig_d::MatPrec ValueType;
        typedef typename TConfig_d::IndPrec IndexType;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecBool>::Type>  BVector;
        typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecFloat>::Type> FVector;
        typedef typename Matrix<TConfig_d>::MVector VVector;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename Matrix_d::IVector IVector;
        typedef Vector<TConfig_d> Vector_d;


    public:
        CR_Selector(AMG_Config &cfg, const std::string &cfg_scope);

        virtual ~CR_Selector();

    private:
        Solver<TConfig_d> *m_smoother;
        void markCoarseFinePoints_1x1(Matrix_d &A,
                                      IVector &cf_map);

        void presmoothFineError(Matrix_d &A, const VVector &AdiagValues, const IVector &cf_map,
                                Vector_d &v_u, Vector_d &v_tmp, Vector_d &v_z,
                                Solver<TConfig_d> *smoother, ValueType &norm0,
                                const ValueType rho_thresh, const int pre);

        void CR_iteration(Matrix_d &A, IVector &cf_map, int &numFine,
                          Vector_d &v_err, ValueType &norm0,
                          const ValueType maxit);
};

template<class T_Config>
class CR_SelectorFactory : public SelectorFactory<T_Config>
{
    public:
        Selector<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) 
        { 
          return new CR_Selector<T_Config>(cfg, cfg_scope); 
        }

};

} // namespace classical
} // namespace amgx

