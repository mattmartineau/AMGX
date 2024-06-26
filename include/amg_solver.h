// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <string>
#include <iostream>

#ifdef _WIN32
#pragma warning (push)
#pragma warning (disable : 4244 4267 4521)
#endif

#ifdef _WIN32
#pragma warning (pop)
#endif

#include <fstream>
#include <limits>

#include <vector.h>
#include <matrix.h>
#include <basic_types.h>
#include <types.h>
#include <misc.h>

#include <amg_config.h>
#include <resources.h>
#include <thread_manager.h>

#include <error.h>

#include <memory>

#include <amgx_types/util.h>

namespace amgx
{

template <class T_Config> class AMG_Solver;
template <class T_Config> class Solver;

AMGX_ERROR initialize();
void finalize();

class AMG_Config;
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec> class AMG;

template <class T_Config>
class AMG_Solver
{
        static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
        static const AMGX_MatPrecision matPrec = T_Config::matPrec;
        static const AMGX_IndPrecision indPrec = T_Config::indPrec;
        typedef TemplateConfig<AMGX_host, vecPrec, matPrec, indPrec> T_Config_h;
        typedef TemplateConfig<AMGX_device, vecPrec, matPrec, indPrec> T_Config_d;
        typedef Matrix<T_Config_h> Matrix_h;
        typedef Matrix<T_Config_d> Matrix_d;

        typedef Vector<T_Config_h> Vector_h;
        typedef Vector<T_Config_d> Vector_d;

        typedef typename T_Config_h::MatPrec ValueTypeA;
        typedef typename T_Config_h::VecPrec ValueTypeB;

        typedef typename T_Config_h::template setVecPrec< types::PODTypes< ValueTypeB >::vec_prec >::Type PODConfig_h;
        typedef typename T_Config_d::template setVecPrec< types::PODTypes< ValueTypeB >::vec_prec >::Type PODConfig_d;

        typedef Vector<PODConfig_h> PODVector_h;
        typedef Vector<PODConfig_d> PODVector_d;

        typedef AMG<vecPrec, matPrec, indPrec> AMG_Class;

    public:
        AMG_Solver(Resources *res, AMG_Configuration *cfg = NULL);      // new in API v2, grab configuration by the pointer (if NULL - from resources), saves the pointer
        AMG_Solver(Resources *res, AMG_Configuration &cfg);           // external configuration, saves the copy
        AMG_Solver(const AMG_Solver<T_Config>  &amg_solver);
        AMG_Solver &operator=(const AMG_Solver &amg_solver);
        ~AMG_Solver();

        /****************************************************
        * Sets A as the matrix for the AMG system
        ****************************************************/
        AMGX_ERROR setup( Matrix<T_Config> &A );

        /****************************************************
        * Sets A as the matrix for the AMG system
        ****************************************************/
        AMGX_ERROR resetup( Matrix<T_Config> &A );
        AMGX_ERROR setup_capi( std::shared_ptr<Matrix<T_Config>> pA0);
        AMGX_ERROR resetup_capi( std::shared_ptr<Matrix<T_Config>> pA0);

        /****************************************************
        * Solves the AMG system Ax=b.
        ***************************************************/
        AMGX_ERROR solve( Vector<T_Config> &b, Vector<T_Config> &x, AMGX_STATUS &status, bool xIsZero = false );

        const PODVector_h &get_residual( int res_num ) const;
        int get_num_iters();

        Solver<T_Config> *getSolverObject( ) { return solver; }
        const Solver<T_Config> *getSolverObject( ) const { return solver; }

        inline Resources *getResources() const { return m_resources; }
        inline AMG_Config *getConfig() const { return m_cfg; }
        inline void setResources(Resources *resources) { m_resources = resources; }

        int getStructureReuseLevels();

    private:
        void process_config(AMG_Config &in_cfg, std::string solver_scope);

        void init();

        AMG_Config *m_cfg;
        bool m_cfg_self;
        Resources *m_resources;
        Solver<T_Config> *solver;

        int ref_count;

        // reusing matrix structure
        std::string structure_reuse_levels_scope;

        // Do we include timings.
        bool m_with_timings;
        cudaEvent_t m_setup_start, m_setup_stop;
        cudaEvent_t m_solve_start, m_solve_stop;

        Matrix<T_Config> &get_A(void)
        {
            return *m_ptrA;
        }

        std::shared_ptr<Matrix<T_Config>> m_ptrA;

        void mem_manage(Matrix<T_Config> &A)
        {
            m_ptrA.reset(new Matrix<T_Config>(A));
        }
};

} // namespace amgx
