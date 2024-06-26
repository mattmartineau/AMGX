// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


namespace amgx
{
template<class TConfig> class Convergence;
}

#include <amg.h>

#include "amgx_types/util.h"

namespace amgx
{

template< typename T >
struct Epsilon_conv
{};

template<>
struct Epsilon_conv<float>
{
    static __device__ __host__ __forceinline__ float value( ) { return 1.0e-6f; }
};

template<>
struct Epsilon_conv<double>
{
    static __device__ __host__ __forceinline__ double value( ) { return 1.0e-12; }
};

template<>
struct Epsilon_conv<cuComplex>
{
    static __device__ __host__ __forceinline__ float value( ) { return 1.0e-6f; }
};

template<>
struct Epsilon_conv<cuDoubleComplex>
{
    static __device__ __host__ __forceinline__ double value( ) { return 1.0e-12; }
};

inline bool isConverged(AMGX_STATUS const &conv_stat)
{
    return conv_stat == AMGX_ST_CONVERGED;
}

inline bool isDiverged(AMGX_STATUS const &conv_stat)
{
    return conv_stat == AMGX_ST_DIVERGED;
}

inline bool isDone(AMGX_STATUS const &conv_stat)
{
    return ( conv_stat != AMGX_ST_NOT_CONVERGED );
}

template<class TConfig>
class Convergence
{
    public:
        typedef typename TConfig::MatPrec ValueTypeA;
        typedef typename TConfig::VecPrec ValueTypeB;
        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef typename TConfig::template setMemSpace<AMGX_device>::Type TConfig_d;
        typedef Vector<typename TConfig::template setVecPrec<types::PODTypes<ValueTypeB>::vec_prec>::Type> PODVec;
        typedef Vector<typename TConfig_h::template setVecPrec<types::PODTypes<ValueTypeB>::vec_prec>::Type> PODVec_h;

        typedef Vector<TConfig_h> Vector_h;
        typedef Vector<TConfig_d> Vector_d;
        typedef Vector<TConfig> VVector;
        Convergence(AMG_Config &cfg, const std::string &cfg_scope);
        virtual ~Convergence() {};

        // Sets the solver name
        inline void setName(std::string &convergence_name) { m_convergence_name = convergence_name; }

        // Returns the name of the solver
        inline std::string getName() const { return m_convergence_name; }

        // Initialize the before running the iterations.
        virtual void convergence_init();

        // Run a single iteration. Compute the residual and its norm and decide convergence.
        virtual AMGX_STATUS convergence_update_and_check(const PODVec_h &nrm, const PODVec_h &nrm_ini);

        // Define the tolerance. Does nothing if tolerance doesn't make sense.
        void setTolerance( double tol ) { m_tolerance = tol; }

    protected:
        std::string m_convergence_name;
        AMG_Config *m_cfg;
        std::string m_cfg_scope;

        double m_tolerance;
};

template<class TConfig>
class ConvergenceFactory
{
    public:
        virtual Convergence<TConfig> *create(AMG_Config &cfg, const std::string &cfg_scope) = 0;
        virtual ~ConvergenceFactory() {};

        /********************************************
         * Register a convergence class with key "name"
         *******************************************/
        static void registerFactory(std::string name, ConvergenceFactory<TConfig> *f);

        /********************************************
         * Unregister a convergence class with key "name"
         *******************************************/
        static void unregisterFactory(std::string name);

        /********************************************
         * Unregister all the solver classes
         *******************************************/
        static void unregisterFactories( );

        /*********************************************
        * Allocates solvers based on cfg
        *********************************************/
        static Convergence<TConfig> *allocate(AMG_Config &cfg, const std::string &current_scope);

        typedef typename std::map<std::string, ConvergenceFactory<TConfig>*>::const_iterator Iterator;

        static Iterator getIterator() { return getFactories().begin(); };
        static bool isIteratorLast(const Iterator &iter) { if ( iter == getFactories().end() ) return true; else return false; };

    private:
        static std::map<std::string, ConvergenceFactory<TConfig>*> &getFactories( );
};

} // namespace amgx
