#pragma once

// =====================================================================================
// THE AGGREGATION OF NEAR-COINCIDENT SEEDS ( `solvers_des_familles` README § 23.3 - § 23.12 ), the same for the CPU solver
// ( `Newton.h`, `Solve.h` ) and the card's ( `gpu/Newton2D.cuh`, whose host code drives it ).
//
// = The floor, and the criterion
//
// Two seeds `delta` apart are separated by a plane whose offset is `( w_i - w_j ) / ( 2 delta )`: the mass that moves between
// them per unit of weight difference is the facet's coefficient of the laplacian, `c_ij = int_{facet} rho / ( 2 delta )`. The
// weights are doubles, so `w_i - w_j` is resolved to `ulp( max( |w_i|, |w_j| ) )`, and no weight vector places that plane
// better than `m_ij = c_ij ulp( w )` of mass ( `lines_equal`: a pair 1.009e-8 apart, `m = 5.4e-11`, 5.4e-6 of its target --
// the 2.35e-6 the solve stagnated at ). So the pair is MERGED when
//
//     m_ij = c_ij ulp( max( |w_i|, |w_j| ) )  >  tau_ij / margin,      tau_ij = max( tol_abs, tol_rel min( nu_i, nu_j ) )
//
// i.e. when its floor is within `margin` of the tolerance asked for ( a pair lands at best within half an ulp step; measured
// 0.43 of it on `lines_equal` -- `margin = 4` leaves room for Newton's own landing ). `tau` is never taken below
// `TAU_REL_MIN nu` ( 1e-12 ): a tolerance under that is not one the solver serves ( Newton's last step lands between 1e-11
// and 1e-16 of a cell's mass on the bench clouds, the measures' own rounding is ~1e-15 ), and without the bound a request at
// 1e-15 merged ordinary pairs whose floor is 1e-14 ( 1259 seeds of a uniform cloud of 3000 ). Both factors are read ON THE DIAGRAM:
// `c_ij` is the facet's coefficient, which carries the local spacing ( measure of the facet over the distance -- the long
// facets of a stretched cell included, which no estimate from the positions alone gets right ) and the density; the weights
// are those of the iterate. The facets of a diagram contain every close pair ( the Delaunay graph contains the euclidean
// minimum spanning tree, § 23.8 ), so the detection needs no other structure: a scan of the coefficients, done only when
// the largest diagonal of the laplacian passes `tau_min / ( margin ulp( max |w| ) )` -- never on a cloud without such a pair
// ( uniform 1e5: the diagonal is ~1e3, the bound ~1e9 ). Merges are latched: a cluster only grows.
//
// EXACT DUPLICATES are not seen by the diagram ( no plane between two equal points: each one's cell is the whole merged
// cell, the masses count it twice ). They are given by the caller ( `SdotPlanNd`, a hash of the positions ), and kept
// EMPTY: `w_dup = w_rep - k gap` for the `k`-th duplicate of a representative ( any gap > 0 empties the cell of a duplicate
// whatever the iterate; `gap` is far above the rounding of the cells' cuts, and two duplicates of one point never share a
// weight: the generic 3D cell mistakes the plane `0 . x <= 0` ), the representative ( the smallest index ) carrying the
// targets of its duplicates.
//
// = What is solved
//
// The aggregated problem on the TRUE diagram ( not on a cloud whose clusters are moved to their barycentres: § 23.10 measured
// what moving them costs the neighbours, 4.3e-2 on a stretched cell, which only a Newton on the whole cloud could undo ): the
// tests ( the stopping test, the merit of the damping ) read the masses of the clusters, each member's share being its part
// of the cluster's mass in proportion to its target ( `a~_i = nu_i a_r / nu_r` ); the direction is still the full Newton
// direction, which places the planes inside a cluster as well as the doubles let it. Once the aggregated problem has
// converged, THE RE-SPLITTING: the local problem of each merged cell, its members' weights alone, the rest of the diagram
// fixed -- a few Newton steps on a `k x k` system ( the cluster's block of the laplacian ), one diagram each, kept while the
// members' residual decreases. What remains is the floor `m_ij`, below which the weights cannot go ( § 23.11 ).
// =====================================================================================

#include "Laplacian.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace sdot {
namespace sdotplan {

/// the spacing of the doubles at `x` ( `ulp( 0 )`: the smallest denormal )
inline double ulp_of( double x ) {
    x = std::fabs( x );
    return std::nextafter( x, INFINITY ) - x;
}

/// the smallest relative tolerance the criterion reads ( see the head of the file )
constexpr double TAU_REL_MIN = 1e-12;

struct Aggregation {
    // ---- the settings ( `SdotPlanNd` )
    double margin  = 0;          ///< `kappa` of the criterion ( 0: no merge of near-coincident seeds )
    double gap     = 0;          ///< exact duplicates: `w_dup = w_rep - gap`
    double tol_abs = 0, tol_rel = 0;   ///< the solve's stopping test ( the criterion's `tau` )

    // ---- the state
    SI                  n = 0;
    std::vector<SI>     parent;      ///< union-find over the seeds ( empty: nothing merged )
    std::vector<SI>     dup_rep;     ///< exact duplicate: its representative, -1 otherwise ( empty: none )
    std::vector<SI>     dups;        ///< the exact duplicates ( not their representatives )
    std::vector<double> dup_gap;     ///< ... `w_rep - w_dup` for each
    std::vector<SI>     cl_begin;    ///< the clusters, CSR ( `cl_begin.size() - 1` clusters )
    std::vector<SI>     cl_members;  ///< ... the members of each, increasing ( the representative first )
    std::vector<SI>     cluster_of;  ///< -1 or the cluster of a seed
    std::vector<double> nu_e;        ///< the target each seed answers for ( a representative: its duplicates' too; a duplicate: 0 )
    double              tau_min = 0; ///< the smallest `tau_i` ( `prepare` )
    SI                  nb_merged_pairs = 0;

    void init( SI nb ) { n = nb; }

    SI   nb_clusters() const { return cl_begin.empty() ? 0 : SI( cl_begin.size() ) - 1; }
    bool any() const { return nb_clusters() > 0; }
    bool has_dups() const { return ! dups.empty(); }
    bool is_dup( SI i ) const { return ! dup_rep.empty() && dup_rep[ i ] >= 0; }
    SI   nb_aggregated() const { return SI( cl_members.size() ); }

    SI find( SI i ) {
        while ( parent[ i ] != i ) { parent[ i ] = parent[ parent[ i ] ]; i = parent[ i ]; }
        return i;
    }
    /// `true` if `i` and `j` were in two clusters
    bool unite( SI i, SI j ) {
        if ( parent.empty() ) {
            parent.resize( n );
            for ( SI k = 0; k < n; ++k ) parent[ k ] = k;
        }
        i = find( i ); j = find( j );
        if ( i == j ) return false;
        parent[ std::max( i, j ) ] = std::min( i, j );
        return true;
    }

    /// the exact duplicates, `( dup, rep )` pairs ( `rep < dup` ): static clusters
    void set_duplicates( const std::vector<SI> &dup, const std::vector<SI> &rep ) {
        if ( dup.empty() ) return;
        dup_rep.assign( n, -1 );
        std::vector<SI> count( n, 0 );
        for ( size_t k = 0; k < dup.size(); ++k ) {
            dup_rep[ dup[ k ] ] = rep[ k ];
            dups.push_back( dup[ k ] );
            dup_gap.push_back( gap * double( ++count[ rep[ k ] ] ) );
            unite( dup[ k ], rep[ k ] );
        }
        rebuild();
    }

    /// the clusters from `parent`
    void rebuild() {
        cl_begin.clear(); cl_members.clear();
        cluster_of.assign( n, -1 );
        if ( parent.empty() ) return;
        std::vector<SI> size( n, 0 );
        for ( SI i = 0; i < n; ++i ) ++size[ find( i ) ];
        std::vector<SI> id( n, -1 );
        cl_begin.push_back( 0 );
        for ( SI i = 0; i < n; ++i )                     // numbered by their representative ( the smallest index )
            if ( parent[ i ] == i && size[ i ] > 1 ) { id[ i ] = SI( cl_begin.size() ) - 1; cl_begin.push_back( cl_begin.back() + size[ i ] ); }
        cl_members.resize( cl_begin.back() );
        std::vector<SI> at( cl_begin.begin(), cl_begin.end() - 1 );
        for ( SI i = 0; i < n; ++i ) {
            const SI c = id[ find( i ) ];
            if ( c < 0 ) continue;
            cluster_of[ i ] = c;
            cl_members[ at[ c ]++ ] = i;
        }
    }

    /// what depends on the target of the stage: `nu_e`, `tau_min`
    void prepare( const std::vector<double> &nu ) {
        double nu_min = nu.empty() ? 0 : nu[ 0 ];
        for ( double v : nu ) nu_min = std::min( nu_min, v );
        tau_min = tau( nu_min );
        if ( dups.empty() ) { nu_e.clear(); return; }
        nu_e = nu;
        for ( SI i : dups ) { nu_e[ dup_rep[ i ] ] += nu[ i ]; nu_e[ i ] = 0; }
    }

    /// the residual a seed of target `nu_i` must reach, as the criterion reads it
    double tau( double nu_i ) const { return std::max( { tol_abs, tol_rel * nu_i, TAU_REL_MIN * nu_i } ); }

    /// the target the tests and the right-hand side read for each seed
    const std::vector<double> &targets( const std::vector<double> &nu ) const { return dups.empty() ? nu : nu_e; }

    /// `w_dup = w_rep - k gap` ( the representatives' weights are the solved ones )
    void tie( std::vector<double> &w ) const {
        for ( size_t k = 0; k < dups.size(); ++k ) w[ dups[ k ] ] = w[ dup_rep[ dups[ k ] ] ] - dup_gap[ k ];
    }

    /// THE LINEAR SYSTEM'S ROWS OF THE DUPLICATES: an empty cell has no facet, and an isolated row ( its diagonal set to one )
    /// breaks the solvers that take the kernel of the laplacian for the constants ( the multigrid: 1e5 iterations in 3D ). Each
    /// duplicate is LINKED to its representative by a facet of the representative's own scale ( the sum of its row ): the
    /// duplicate's equation is then `d_dup = d_rep` ( its right-hand side is zero ), and the representative's is unchanged --
    /// the same direction as without the duplicate, whatever the coefficient. Appended to a copy of the facets.
    void link_duplicates( const std::vector<Facet> &fa, std::vector<Facet> &out ) const {
        out = fa;
        std::vector<double> sum( n, 0.0 );
        for ( const Facet &e : fa ) sum[ e.i ] += e.c;
        for ( SI i : dups ) {
            const SI r = dup_rep[ i ];
            const double c = sum[ r ] > 0 ? sum[ r ] : 1.0;
            out.push_back( Facet{ i, r, c } );
            out.push_back( Facet{ r, i, c } );
        }
    }

    /// the masses the tests read: each member of a cluster gets its share of the cluster's mass, `nu_i a_r / nu_r`
    const std::vector<double> &shares( const std::vector<double> &a, const std::vector<double> &nu, std::vector<double> &ae ) const {
        if ( ! any() ) return a;
        ae = a;
        for ( SI c = 0; c < nb_clusters(); ++c ) {
            double sa = 0, sn = 0;
            for ( SI k = cl_begin[ c ]; k < cl_begin[ c + 1 ]; ++k ) { sa += a[ cl_members[ k ] ]; sn += nu[ cl_members[ k ] ]; }
            const double r = sn > 0 ? sa / sn : 0;
            for ( SI k = cl_begin[ c ]; k < cl_begin[ c + 1 ]; ++k ) ae[ cl_members[ k ] ] = nu[ cl_members[ k ] ] * r;
        }
        return ae;
    }

    /// the smallest mass the floor of the damping reads ( the duplicates, empty by construction, left out )
    double floor_min( const std::vector<double> &a ) const {
        double m = 1e300;
        if ( dups.empty() ) {
            for ( double x : a ) m = std::min( m, x );
            return a.empty() ? 0 : m;
        }
        for ( SI i = 0; i < SI( a.size() ); ++i )
            if ( dup_rep[ i ] < 0 ) m = std::min( m, a[ i ] );
        return m;
    }

    /// THE RESIDUAL OF THE FULL PROBLEM: `max |a_i - nu_e_i|` and its relative form, the duplicates left out ( no weight
    /// vector splits a cell between two equal points: their representative answers for them )
    void full_residual( const std::vector<double> &a, const std::vector<double> &nu, double &worst, double &worst_rel ) const {
        const std::vector<double> &N = targets( nu );
        worst = worst_rel = 0;
        for ( SI i = 0; i < SI( a.size() ); ++i ) {
            if ( is_dup( i ) ) continue;
            const double e = std::fabs( a[ i ] - N[ i ] );
            worst = std::max( worst, e );
            worst_rel = std::max( worst_rel, e / N[ i ] );
        }
    }

    bool converged( double worst, double worst_rel ) const {
        return worst <= tol_abs || ( tol_rel > 0 && worst_rel <= tol_rel );
    }

    /// THE DETECTION ( see the head of the file ) on the laplacian of the accepted diagram and its weights; `true` if a
    /// cluster was made or grew. The scan of the coefficients is skipped while no diagonal can hold one above the bound.
    bool detect( const Laplacian &L, const std::vector<double> &w, const std::vector<double> &nu ) {
        if ( ! ( margin > 0 ) || ! ( tau_min > 0 ) ) return false;
        double wmax = 0, dmax = 0;
        for ( double x : w ) wmax = std::max( wmax, std::fabs( x ) );
        const double c_star = tau_min / ( margin * ulp_of( wmax ) );
        for ( SI i = 0; i < L.n; ++i ) dmax = std::max( dmax, L.dia[ i ] );
        if ( ! ( dmax > c_star ) ) return false;
        bool changed = false;
        for ( SI i = 0; i < L.n; ++i ) {
            if ( ! ( L.dia[ i ] > c_star ) ) continue;
            for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) {
                const SI j = L.col[ e ];
                if ( j <= i || ! ( L.c[ e ] > c_star ) ) continue;
                const double m = L.c[ e ] * ulp_of( std::max( std::fabs( w[ i ] ), std::fabs( w[ j ] ) ) );
                if ( m * margin > tau( std::min( nu[ i ], nu[ j ] ) ) && unite( i, j ) ) { changed = true; ++nb_merged_pairs; }
            }
        }
        if ( changed ) rebuild();
        return changed;
    }

    /// THE RE-SPLITTING, one Newton step of the local problem of each merged cell: `dw` ( size `n`, zero outside the
    /// clusters ) solves `L_CC dw_C = nu_C - a_C` on the members of each cluster ( the duplicates left out ), `L_CC` the block
    /// of the laplacian `L` ( its diagonal is the whole row: the facets with the neighbours count, the neighbours do not
    /// move ). A cluster of more than `k_max` members is left as it is. Returns the members' largest residual before the step.
    double local_step( const Laplacian &L, const std::vector<double> &a, const std::vector<double> &nu, std::vector<double> &dw, SI k_max = 64 ) const {
        const std::vector<double> &N = targets( nu );
        dw.assign( n, 0.0 );
        double worst = 0;
        std::vector<SI> mem;
        std::vector<double> M, r;
        for ( SI c = 0; c < nb_clusters(); ++c ) {
            movable_members( c, mem );
            const SI k = SI( mem.size() );
            for ( SI i : mem ) worst = std::max( worst, std::fabs( a[ i ] - N[ i ] ) );
            if ( k < 2 || k > k_max ) continue;
            M.assign( size_t( k * k ), 0.0 );
            r.resize( k );
            for ( SI u = 0; u < k; ++u ) {
                const SI i = mem[ u ];
                M[ u * k + u ] = L.dia[ i ];
                r[ u ] = N[ i ] - a[ i ];
                for ( SI e = L.row[ i ]; e < L.row[ i + 1 ]; ++e ) {
                    const auto it = std::lower_bound( mem.begin(), mem.end(), L.col[ e ] );
                    if ( it != mem.end() && *it == L.col[ e ] ) M[ u * k + SI( it - mem.begin() ) ] -= L.c[ e ];
                }
            }
            if ( ! cholesky_solve( k, M, r ) ) continue;
            for ( SI u = 0; u < k; ++u ) dw[ mem[ u ] ] = r[ u ];
        }
        return worst;
    }

    /// `M x = r` for the `k x k` symmetric positive definite `M` ( row-major, overwritten ): `x` in `r`. `false` if `M` is not
    /// positive definite ( a cluster without a neighbour )
    static bool cholesky_solve( SI k, std::vector<double> &M, std::vector<double> &r ) {
        for ( SI p = 0; p < k; ++p ) {
            double s = M[ p * k + p ];
            for ( SI q = 0; q < p; ++q ) s -= M[ p * k + q ] * M[ p * k + q ];
            if ( ! ( s > 0 ) ) return false;
            M[ p * k + p ] = std::sqrt( s );
            for ( SI u = p + 1; u < k; ++u ) {
                double t = M[ u * k + p ];
                for ( SI q = 0; q < p; ++q ) t -= M[ u * k + q ] * M[ p * k + q ];
                M[ u * k + p ] = t / M[ p * k + p ];
            }
        }
        for ( SI p = 0; p < k; ++p ) {                    // L y = r, then L^t x = y
            double s = r[ p ];
            for ( SI q = 0; q < p; ++q ) s -= M[ p * k + q ] * r[ q ];
            r[ p ] = s / M[ p * k + p ];
        }
        for ( SI p = k - 1; p >= 0; --p ) {
            double s = r[ p ];
            for ( SI q = p + 1; q < k; ++q ) s -= M[ q * k + p ] * r[ q ];
            r[ p ] = s / M[ p * k + p ];
        }
        return true;
    }

    /// the members of cluster `c` that the re-splitting moves ( the duplicates left out ), increasing
    void movable_members( SI c, std::vector<SI> &mem ) const {
        mem.clear();
        for ( SI k = cl_begin[ c ]; k < cl_begin[ c + 1 ]; ++k )
            if ( ! is_dup( cl_members[ k ] ) ) mem.push_back( cl_members[ k ] );
    }

    /// the members' largest residual ( the duplicates left out )
    double members_residual( const std::vector<double> &a, const std::vector<double> &nu ) const {
        const std::vector<double> &N = targets( nu );
        double worst = 0;
        for ( SI i : cl_members )
            if ( ! is_dup( i ) ) worst = std::max( worst, std::fabs( a[ i ] - N[ i ] ) );
        return worst;
    }
};

} // namespace sdotplan
} // namespace sdot
