#pragma once

// =====================================================================================
// THE LAPLACIAN OF THE LAGUERRE GRAPH, assembled from the facets ( `solvers_des_familles`,
// `src/solver/Laplacian.h`, taken over as is ).
//
//     c_ij = int_{facet ij} rho / ( 2 |p_i - p_j| ),   L_ii = sum_j c_ij,   L_ij = -c_ij
//
// The formula does not depend on the dimension: length over distance in 2D, area over distance in
// 3D, and it is the cell that delivers the facet's measure. Its kernel is exactly the
// CONSTANTS -- adding the same constant to all weights changes no cell.
//
// WITHOUT A SINGLE SORT. Each facet is seen TWICE, once per cell, and the two measures only
// differ by rounding; we keep the one of the row, so a count and a prefix sum
// place everyone -- and each row sums EXACTLY to zero. Pairing the two measures with
// a sort cost more than the diagram itself ( 4.2 s against 3.9 at n=1e6 ).
//
// THE GAUGE `w_0 = 0`: the reduced system strikes out row and column 0, and becomes positive definite.
// `reduced_crs` delivers it that way, columns sorted, for a solver that wants CRS.
// =====================================================================================

#include <loom/support/common_types.h>
#include <vector>

namespace sdot {
namespace sdotplan {

/// A facet seen FROM cell `i`: it is the one that measured its extent.
struct Facet {
    SI     i, j;
    double c;                      ///< `c_ij > 0`; the minus sign is in the matrix
};

struct Laplacian {
    SI                  n = 0;
    std::vector<SI>     row, col;  ///< CSR of the off-diagonals, `c[ k ] = c_ij`
    std::vector<double> c;
    std::vector<double> dia;       ///< `L_ii`, the sum of the row

    void assemble( SI nb, const std::vector<Facet> &fa ) {
        n = nb;
        row.assign( n + 1, 0 );
        for ( const Facet &e : fa )
            ++row[ e.i + 1 ];
        for ( SI i = 0; i < n; ++i )
            row[ i + 1 ] += row[ i ];

        col.resize( row[ n ] );
        c.resize( row[ n ] );
        dia.assign( n, 0.0 );
        std::vector<SI> at( row.begin(), row.end() - 1 );
        for ( const Facet &e : fa ) {
            const SI p = at[ e.i ]++;
            col[ p ] = e.j;
            c[ p ] = e.c;
            dia[ e.i ] += e.c;
        }
        // a cell without a neighbor cannot happen as long as none is empty, but a zero row
        // would make the system singular WITHOUT SAYING SO: we neutralize it.
        for ( SI i = 0; i < n; ++i )
            if ( ! ( dia[ i ] > 0 ) )
                dia[ i ] = 1;
    }

    /// `y = L x` ( the full system, gauge included )
    void product( const double *x, double *y ) const {
        for ( SI i = 0; i < n; ++i ) {
            double s = dia[ i ] * x[ i ];
            for ( SI e = row[ i ]; e < row[ i + 1 ]; ++e )
                s -= c[ e ] * x[ col[ e ] ];
            y[ i ] = s;
        }
    }

    /// THE REDUCED SYSTEM ( `n - 1` unknowns, seed 0 struck out -- not neutralized: an identity row
    /// would give the aggregation an isolated node to deal with ), diagonal included, columns
    /// SORTED by insertion -- seven entries per row.
    void reduced_crs( std::vector<int> &ptr, std::vector<int> &cl, std::vector<double> &val ) const {
        const SI m = n - 1;
        ptr.assign( m + 1, 0 );
        for ( SI i = 1; i < n; ++i ) {
            int k = 1;                                  // the diagonal, always present
            for ( SI e = row[ i ]; e < row[ i + 1 ]; ++e )
                k += col[ e ] >= 1;
            ptr[ i ] = k;
        }
        for ( SI i = 0; i < m; ++i )
            ptr[ i + 1 ] += ptr[ i ];
        cl.resize( ptr[ m ] );
        val.resize( ptr[ m ] );
        for ( SI i = 1; i < n; ++i ) {
            int k = ptr[ i - 1 ];
            cl[ k ] = int( i - 1 );  val[ k ] = dia[ i ];  ++k;
            for ( SI e = row[ i ]; e < row[ i + 1 ]; ++e )
                if ( col[ e ] >= 1 ) {
                    cl[ k ] = int( col[ e ] - 1 );  val[ k ] = -c[ e ];  ++k;
                }
            for ( int u = ptr[ i - 1 ] + 1; u < k; ++u ) {
                const int cc = cl[ u ];
                const double v = val[ u ];
                int j = u;
                for ( ; j > ptr[ i - 1 ] && cl[ j - 1 ] > cc; --j ) {
                    cl[ j ] = cl[ j - 1 ]; val[ j ] = val[ j - 1 ];
                }
                cl[ j ] = cc; val[ j ] = v;
            }
        }
    }
};

} // namespace sdotplan
} // namespace sdot
