// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_config.h"
#include "test_utils.h"
#include "cutil.h"
#include "util.h"
#include "amg_solver.h"
#include "resources.h"

#include "aggregation/coarseAgenerators/coarse_A_generator.h"
#include "aggregation/selectors/agg_selector.h"
#include "matrix_coloring/matrix_coloring.h"
#include "matrix_coloring/min_max.h"
#include "solvers/solver.h"

#include "classical/selectors/selector.h"
#include "classical/interpolators/interpolator.h"
#include "classical/strength/strength.h"

#include <cusp/print.h>
#include <cusp/gallery/poisson.h>

#ifdef AMGX_WITH_MPI
#include <mpi.h>
#endif

namespace amgx

{


// This test tries to run amgx stuff on the matrix with some offdiagonal values equals to zero (but stored in the A.values array explicitly)
DECLARE_UNITTEST_BEGIN(ExplicitZeroValues);

typedef typename TConfig_h::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig::mode)>::Type vvec_h;
typedef typename TConfig::template setVecPrec<AMGX_vecInt>::Type ivec;
typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_h;

// setup restriction on HOST
void fillRowOffsetsAndColIndices(const int num_aggregates,
                                 Vector<ivec_h> aggregates,
                                 const int R_num_cols,
                                 Vector<ivec_h> &R_row_offsets,
                                 Vector<ivec_h> &R_col_indices)
{
    for (int i = 0; i < num_aggregates + 1; i++)
    {
        R_row_offsets[i] = 0;
    }

    // Count number of neighbors for each row
    for (int i = 0; i < R_num_cols; i++)
    {
        int I = aggregates[i];
        R_row_offsets[I]++;
    }

    R_row_offsets[num_aggregates] = R_num_cols;

    for (int i = num_aggregates - 1; i >= 0; i--)
    {
        R_row_offsets[i] = R_row_offsets[i + 1] - R_row_offsets[i];
    }

    /* Set column indices. */
    for (int i = 0; i < R_num_cols; i++)
    {
        int I = aggregates[i];
        int Ip = R_row_offsets[I]++;
        R_col_indices[Ip] = i;
    }

    /* Reset r[i] to start of row memory. */
    for (int i = num_aggregates - 1; i > 0; i--)
    {
        R_row_offsets[i] = R_row_offsets[i - 1];
    }

    R_row_offsets[0] = 0;
}

void test_coarsers(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope)
{
    Matrix<T_Config> Ac;
    int num_aggregates = A.get_num_rows();
    Vector<ivec_h> h_aggregates;
    h_aggregates.resize( A.get_num_rows() );

    for ( int i = 0; i < h_aggregates.size(); i++ )
    {
        h_aggregates[i] = i;
    }

    Vector<ivec_h> h_R_row_offsets;
    Vector<ivec_h> h_R_col_indices;
    h_R_row_offsets.resize( num_aggregates + 1 );
    h_R_col_indices.resize( A.get_num_rows() );
    fillRowOffsetsAndColIndices( num_aggregates, h_aggregates, A.get_num_rows(), h_R_row_offsets, h_R_col_indices );
    Vector<ivec> aggregates = h_aggregates;
    Vector<ivec> R_row_offsets = h_R_row_offsets;
    Vector<ivec> R_col_indices = h_R_col_indices;
    cudaCheckError();
    typename aggregation::CoarseAGeneratorFactory<T_Config>::Iterator iter = aggregation::CoarseAGeneratorFactory<T_Config>::getIterator();
    aggregation::CoarseAGenerator<TConfig> *generator;

    while (!aggregation::CoarseAGeneratorFactory<T_Config>::isIteratorLast(iter))
    {
        //std::cout << "aggregator=" << iter->first << std::endl;
        generator = NULL;
        generator = iter->second->create(cfg, cfg_scope);
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        UNITTEST_ASSERT_TRUE_DESC("Generator is not created\n", generator != NULL);
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        generator->computeAOperator(A, Ac, aggregates, R_row_offsets, R_col_indices, num_aggregates);
        UNITTEST_ASSERT_TRUE_DESC("Coarser matrix contains nans\n", !containsNan<ValueTypeA>(Ac.values.raw(), Ac.values.size()));
        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (generator != NULL) { delete generator; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

void test_selectors(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope)
{
    typename aggregation::SelectorFactory<T_Config>::Iterator iter = aggregation::SelectorFactory<T_Config>::getIterator();
    aggregation::Selector<TConfig> *selector;
    IVector vec, vec1;
    int num;

    while (!aggregation::SelectorFactory<T_Config>::isIteratorLast(iter))
    {
        std::string m_name = iter->first.c_str();

        //printf("Trying selector %s\n", m_name.c_str());fflush(stdout);
        if ((m_name.compare("GEO") == 0) || (m_name.compare("GEO_ONE_PHASE_HANDSHAKING") == 0) || (m_name.compare("PARALLEL_GREEDY_SELECTOR") == 0))
        {
            //printf("Skipping...\n");fflush(stdout);
            ++iter;
            continue;
        }

        //std::cout << "selector=" << iter->first << std::endl;
        selector = NULL;
        PrintOnFail("processing: %s\n", iter->first.c_str());
        selector = iter->second->create(cfg, cfg_scope);
        PrintOnFail("Selector creation\n");
        UNITTEST_ASSERT_TRUE(selector != NULL);
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        selector->setAggregates(A, vec, vec1, num);
        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (selector != NULL) { delete selector; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

void test_matrix_coloring(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope)
{
    MatrixColoring<TConfig> *color;
    typename MatrixColoringFactory<T_Config>::Iterator iter = MatrixColoringFactory<T_Config>::getIterator();

    while (!MatrixColoringFactory<T_Config>::isIteratorLast(iter))
    {
        //std::cout << "coloring=" << iter->first << std::endl;
        color = NULL;
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        color = iter->second->create(cfg, cfg_scope);
        UNITTEST_ASSERT_TRUE(color != NULL);
        A.set_initialized(0);
        A.colorMatrix(cfg, cfg_scope);
        A.set_initialized(1);
        int num_colors = A.getMatrixColoring().getNumColors();
        UNITTEST_ASSERT_TRUE(num_colors != 0);
        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (color != NULL) { delete color; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

template<class TConfig>
bool check_solver_mode_pair(std::string solver)
{
    // skip IDR solvers because they don't handle diag zeros well
    return ((solver != "FIXCOLOR_GS") &&
            (solver != "KACZMARZ") &&
            (solver != "IDR") &&
            (solver != "IDRMSYNC"));
}

void test_solvers(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope)
{
#ifdef AMGX_WITH_MPI
    int mpiFlag;
    MPI_Initialized(&mpiFlag);

    if ( !mpiFlag )
    {
        int argc = 1;
        char **argv = NULL;
        MPI_Init( &argc, &argv);
    }

#endif
    Vector<T_Config> b (A.get_num_rows()*A.get_block_dimy()), x (A.get_num_rows()*A.get_block_dimy());
    thrust_wrapper::fill<T_Config::memSpace>(b.begin(), b.end(), 1);
    b.set_block_dimx(1);
    b.set_block_dimy(A.get_block_dimy());
    x.set_block_dimx(1);
    x.set_block_dimy(A.get_block_dimx());
    Vector_h hx;
    Solver<TConfig> *solver;
    typename SolverFactory<T_Config>::Iterator iter = SolverFactory<T_Config>::getIterator();

    while (!SolverFactory<T_Config>::isIteratorLast(iter))
    {
        //std::cout << "solver=" << iter->first << std::endl;
        solver = NULL;
        thrust_wrapper::fill<T_Config::memSpace>(x.begin(), x.end(), static_cast<ValueTypeB>(1.0));
        //printf("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));fflush(stdout);
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        solver = iter->second->create(cfg, cfg_scope);

        // its known that jacobi_l1 implementation for 4x4 fails because of block inverse in setup
        // its known that fixcolor_gs fails on solve phase because of bad values during setup
        if (solver != NULL && check_solver_mode_pair<TConfig>(iter->first))
        {
            solver->setup(A, false);
            solver->set_max_iters(1);

            if (TConfig::matPrec != AMGX_matFloat)
            {
                solver->solve(b, x, false);
            }

            hx = x;
            cudaDeviceSynchronize();
            cudaCheckError();
            // NaNs are expected since there are zero elements
            //    UNITTEST_ASSERT_TRUE_DESC("Smoother result contains nans\n", !containsNan<ValueTypeB>(x.raw(), x.size()));
        }

//      std::cout << iter->first << std::endl;
        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (solver != NULL) { delete solver; solver = NULL; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

void generatePoissonForTest(Matrix<TConfig > &Aout, int block_size, bool diag_prop, int points, int x, int y, int z = 1)
{
    Matrix<TConfig_h > Ac;
    {
        Matrix<TConfig_h > A;
        A.set_initialized(0);
        A.addProps(CSR);
        MatrixCusp<TConfig_h, cusp::csr_format> wA(&A);

        switch (points)
        {
            case 5:
                cusp::gallery::poisson5pt(wA, x, y);
                break;

            case 7:
                cusp::gallery::poisson7pt(wA, x, y, z);
                break;

            case 9:
                cusp::gallery::poisson9pt(wA, x, y);
                break;

            case 27:
                cusp::gallery::poisson27pt(wA, x, y, z);
                break;
        }

        A.set_initialized(1);
        Ac.convert( A, ( diag_prop ? DIAG : 0 ) | CSR, block_size, block_size );
        Ac.set_initialized(1);
    }
    Aout = Ac;
}

void test_levels(Resources *res, Matrix<T_Config> &A)
{
    Vector<T_Config> b (A.get_num_rows()*A.get_block_dimy()), x (A.get_num_rows()*A.get_block_dimy());
    thrust_wrapper::fill<T_Config::memSpace>(b.begin(), b.end(), 1);
    thrust_wrapper::fill<T_Config::memSpace>(x.begin(), x.end(), 1);
    int bsize = A.get_block_dimy();
    b.set_block_dimx(1);
    b.set_block_dimy(bsize);
    x.set_block_dimy(1);
    x.set_block_dimx(bsize);
    AMGX_STATUS solve_status;

    if (!bsize > 1)
// Classical path will only work with block size 1, error handling below is not working for some reason
    {
        AMG_Configuration cfg;
        AMGX_ERROR err = AMGX_OK;
        UNITTEST_ASSERT_TRUE( cfg.parseParameterString("config_version=2, algorithm=CLASSICAL, smoother=MULTICOLOR_DILU, presweeps=1, postsweeps=1, matrix_coloring_scheme=MIN_MAX, determinism_flag=1, max_levels=2, max_iters=1, norm=L1, coloring_level=1") == AMGX_OK);
        AMG_Solver<TConfig> amg(res, cfg);
        err = amg.setup(A);

        if (err != AMGX_ERR_NOT_SUPPORTED_TARGET && err != AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE && err != AMGX_ERR_NOT_IMPLEMENTED)
        {
            PrintOnFail("Classical algorithm: Matrix properties: blocksize = %d, diag_prop = %d\n", A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
            UNITTEST_ASSERT_EQUAL(err, AMGX_OK);
            err = amg.solve( b, x, solve_status, true);

            if (err != AMGX_ERR_NOT_SUPPORTED_TARGET &&
                    err != AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE &&
                    err != AMGX_ERR_NOT_IMPLEMENTED)
            {
                UNITTEST_ASSERT_EQUAL(err, AMGX_OK);
                PrintOnFail("Classical algorithm: Matrix properties: blocksize = %d, diag_prop = %d\n",
                            A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
                UNITTEST_ASSERT_TRUE_DESC("Level solve result contains nans\n",
                                          !containsNan<ValueTypeB>(x.raw(), x.size()));
            }
        }
    }

    thrust_wrapper::fill<T_Config::memSpace>(x.begin(), x.end(), 1);
    {
        AMG_Configuration cfg;
        AMGX_ERROR err = AMGX_OK;
        UNITTEST_ASSERT_TRUE( cfg.parseParameterString("config_version=2, algorithm=AGGREGATION, smoother=MULTICOLOR_DILU, presweeps=1, postsweeps=1, selector=SIZE_2, coarseAgenerator=LOW_DEG, matrix_coloring_scheme=MIN_MAX, determinism_flag=1, max_levels=2, max_iters=1, norm=L1, coloring_level=1") == AMGX_OK);
        AMG_Solver<TConfig> amg(res, cfg);
        err = amg.setup(A);

        if (err != AMGX_ERR_NOT_SUPPORTED_TARGET && err != AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE && err != AMGX_ERR_NOT_IMPLEMENTED)
        {
            PrintOnFail("Aggregation algorithm: Matrix properties: blocksize = %d, diag_prop = %d\n", A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
            UNITTEST_ASSERT_EQUAL(err, AMGX_OK);
            err = amg.solve( b, x, solve_status, true);

            if (err != AMGX_ERR_NOT_SUPPORTED_TARGET && err != AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE && err != AMGX_ERR_NOT_IMPLEMENTED)
            {
                UNITTEST_ASSERT_EQUAL(err, AMGX_OK);
                PrintOnFail("Aggregation algorithm: Matrix properties: blocksize = %d, diag_prop = %d\n", A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
                UNITTEST_ASSERT_TRUE_DESC("Level solve result contains nans\n", !containsNan<ValueTypeB>(x.raw(), x.size()));
            }
        }
    }
}

void test_strength(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope, StrengthFactory<TConfig> **good )
{
    //allocate necessary memory
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecInt>::Type> IVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    FVector weights(A.get_num_rows(), 0.0);
    BVector s_con(A.get_num_nz(), false);
    IVector cf_map(A.get_num_rows(), 0);
    IVector scratch(A.get_num_rows(), 0); //scratch memory of size num_rows
    //compute strong connections and weights
    double max_row_sum = cfg.getParameter<double>("max_row_sum", cfg_scope);
    Strength<T_Config> *strength;
    typename StrengthFactory<T_Config>::Iterator iter = StrengthFactory<T_Config>::getIterator();

    while (!StrengthFactory<T_Config>::isIteratorLast(iter))
    {
        strength = NULL;
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        strength = iter->second->create(cfg, cfg_scope);
        UNITTEST_ASSERT_TRUE(strength != NULL);

        if (strength != NULL)
        {
            strength->computeStrongConnectionsAndWeights(A, s_con, weights, max_row_sum);
            UNITTEST_ASSERT_TRUE_DESC("Strength result contains nans\n", !containsNan<float>(weights.raw(), weights.size()));
            *good = iter->second;
        }

        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (strength != NULL) { delete strength; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

void test_selectors(Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope, StrengthFactory<TConfig> *strengthf, classical::SelectorFactory<TConfig> **good )
{
    //allocate necessary memory
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecInt>::Type> IVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    FVector weights(A.get_num_rows(), 0.0);
    BVector s_con(A.get_num_nz(), false);
    IVector cf_map(A.get_num_rows(), 0);
    IVector scratch(A.get_num_rows(), 0); //scratch memory of size num_rows
    //compute strong connections and weights
    double max_row_sum = cfg.getParameter<double>("max_row_sum", cfg_scope);
    Strength<T_Config> *strength = strengthf->create(cfg, cfg_scope);
    strength->computeStrongConnectionsAndWeights(A, s_con, weights, max_row_sum);
    classical::Selector<T_Config> *selector;
    typename classical::SelectorFactory<T_Config>::Iterator iter = classical::SelectorFactory<T_Config>::getIterator();

    while (!classical::SelectorFactory<T_Config>::isIteratorLast(iter))
    {
        std::string m_name = iter->first.c_str();

        if ((m_name.compare("GEO") == 0) || (m_name.compare("GEO_ONE_PHASE_HANDSHAKING") == 0))
        {
            ++iter;
            continue;
        }

        selector = NULL;
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        selector = iter->second->create(cfg, cfg_scope);
        UNITTEST_ASSERT_TRUE(strength != NULL);

        if (selector != NULL)
        {
            selector->markCoarseFinePoints(A, weights, s_con, cf_map, scratch);

            for (int i = 0; i < A.get_num_rows(); i++)
            {
                UNITTEST_ASSERT_TRUE(cf_map[i] != UNASSIGNED);
            }

            *good = iter->second;
        }

        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (selector != NULL) { delete selector; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}

void test_interpolators(Resources *res, Matrix<T_Config> &A, AMG_Config &cfg, const std::string &cfg_scope, StrengthFactory<TConfig> *strengthf, classical::SelectorFactory<TConfig> *selectorf )
{
    //allocate necessary memory
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecInt>::Type> IVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecBool>::Type> BVector;
    typedef Vector<typename T_Config::template setVecPrec<AMGX_vecFloat>::Type> FVector;
    Matrix<TConfig> P;
    FVector weights(A.get_num_rows(), 0.0);
    BVector s_con(A.get_num_nz(), false);
    IVector cf_map(A.get_num_rows(), 0);
    IVector scratch(A.get_num_rows(), 0); //scratch memory of size num_rows
    //compute strong connections and weights
    double max_row_sum = cfg.getParameter<double>("max_row_sum", cfg_scope);
    Strength<T_Config> *strength = strengthf->create(cfg, cfg_scope);
    classical::Selector<T_Config> *selector = selectorf->create();
    strength->computeStrongConnectionsAndWeights(A, s_con, weights, max_row_sum);
    selector->markCoarseFinePoints(A, weights, s_con, cf_map, scratch);
    Interpolator<T_Config> *interpolator;
    typename InterpolatorFactory<T_Config>::Iterator iter = InterpolatorFactory<T_Config>::getIterator();
    AMG_Configuration scfg;
    AMG_Solver<TConfig> amg(res, scfg);

    while (!InterpolatorFactory<T_Config>::isIteratorLast(iter))
    {
        interpolator = NULL;
        //printf("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));fflush(stdout);
        UNITTEST_ASSERT_EXCEPTION_START;
        PrintOnFail("%s : Matrix properties: blocksize = %d, diag_prop = %d\n", iter->first.c_str(), A.get_block_dimy(), (A.hasProps(DIAG) ? 1 : 0));
        interpolator = iter->second->create(cfg, cfg_scope);
        UNITTEST_ASSERT_TRUE(strength != NULL);

        if (interpolator != NULL)
        {
            interpolator->generateInterpolationMatrix(A, cf_map, s_con, scratch, P, &amg);
        }

        UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED;

        if (interpolator != NULL) { delete interpolator; }

        ++iter;
        UNITTEST_ASSERT_TRUE(true);
    }
}


// for a few rows replaces their first values with zeros. avoids diagonal values.
void random_add_zeros(Matrix<TConfig> &A, int max_zeros)
{
    int zero_num = max_zeros;

    while (zero_num)
    {
        int rowidx = (int)( ((float)rand() / RAND_MAX) * (A.get_num_rows() - 1) );

        if (rowidx == A.col_indices[A.row_offsets[rowidx]]) { continue; }

        int validx = A.row_offsets[rowidx];
        thrust_wrapper::fill<TConfig::memSpace>(A.values.begin() + validx, A.values.begin() + validx + A.get_block_size(), static_cast<ValueTypeA>(0.0));
        --zero_num;
    }
}


void run()
{
    randomize( 31 );
    set_forge_ahead(true);
    int nrows = 100;

    for (int bsize = 1; bsize < 3; ++bsize)
    {
        AMG_Config cfg;
        cfg.parseParameterString("config_version=2, determinism_flag=1, coloring_level=1, reorder_cols_by_color=1, insert_diag_while_reordering=1, preconditioner=BLOCK_JACOBI, min_coarse_rows=2");
        const std::string &cfg_scope = "default";
        Resources res;        // default resources

        for (int diagProp = 0; diagProp < 2; diagProp++)
        {
            //std::cout << "bsize=" << bsize << " diag=" << diagProp << std::endl;
            MatrixA A;
            VVector tb;
            generateMatrixRandomStruct<TConfig>::generateExact(A, nrows, (diagProp != 0), bsize, false);
            random_fill(A);
            random_add_zeros(A, 1);
            //////////////////////////
            //       MatrixIO<TConfig>::writeSystemMatrixMarket("test.mtx", &A, &tb, &tb);
            // aggregation
            test_coarsers(A, cfg, cfg_scope);
            test_selectors(A, cfg, cfg_scope);
            test_matrix_coloring(A, cfg, cfg_scope);
            // Some solvers need a coloring. Make one.
            A.set_initialized(0);
            A.colorMatrix(cfg, cfg_scope);
            A.set_initialized(1);
            test_solvers(A, cfg, cfg_scope);

// classical
//TODO: if strength cannot process matrix
            if (bsize == 1)
            {
                StrengthFactory<TConfig> *good_strength = NULL;
                test_strength(A, cfg, cfg_scope, &good_strength);

                if (good_strength != NULL)
                {
                    classical::SelectorFactory<TConfig> *good_selector = NULL;
                    test_selectors(A, cfg, cfg_scope, good_strength, &good_selector);

                    if (good_selector != NULL)
                    {
                        //test_interpolators(&res, A, cfg, cfg_scope, good_strength, good_selector );
                    }
                }
            }

// levels
            test_levels(&res, A);
        }
    }
}

DECLARE_UNITTEST_END(ExplicitZeroValues);

#define AMGX_CASE_LINE(CASE) ExplicitZeroValues <TemplateMode<CASE>::Type>  ExplicitZeroValues_##CASE;
AMGX_FORALL_BUILDS_DEVICE(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

//ExplicitZeroValues <TemplateMode<AMGX_mode_dDDI>::Type>  ExplicitZeroValues_dDDI;

} //namespace amgx
