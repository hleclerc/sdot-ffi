#pragma once

#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

/// THE DUMP OF THE FINAL CONNECTIVITY.
///
/// The GPU bench has to replay EXACTLY what the CPU bench replays: the same seeds, the same
/// neighbors, in the same order. Redoing it on its side would mean putting CGAL back in there, so
/// one more dependency and above all a risk of silent divergence. So we write the list once,
/// outside the timing, and both benches read it.
///
/// The file ALSO carries the reference volume computed here in FP64 by `Cell3T<128>`: it is the
/// only witness that lets the GPU bench say whether its FP32 returns the same geometry, cell by
/// cell, and not only in sum -- a sum can be right with wrong cells that compensate each
/// other, we have already seen it on this bench.
///
///   magic "PDC1" | dim | n | nv | own[ n * (dim+1) ] | off[ n+1 ] | nbrs[ nv * (dim+1) ] | ref[ n ]
///
/// All in FP64 on disk: the conversion to FP32 is the subject of the measurement, not of the transport.
inline bool dump_csr( const std::string &path, int dim,
                      const std::vector<double> &own, const std::vector<std::size_t> &off,
                      const std::vector<double> &nbrs, const std::vector<double> &ref ) {
    std::FILE *f = std::fopen( path.c_str(), "wb" );
    if ( ! f ) { std::fprintf( stderr, "dump_csr: %s unreadable\n", path.c_str() ); return false; }
    const std::int32_t magic = 0x31434450;                  // "PDC1"
    const std::int32_t d = dim;
    const std::int64_t n = std::int64_t( off.size() ) - 1;
    const std::int64_t nv = std::int64_t( nbrs.size() ) / ( dim + 1 );
    std::vector<std::int64_t> o( off.begin(), off.end() );
    std::fwrite( &magic, 4, 1, f );
    std::fwrite( &d, 4, 1, f );
    std::fwrite( &n, 8, 1, f );
    std::fwrite( &nv, 8, 1, f );
    std::fwrite( own.data(), 8, own.size(), f );
    std::fwrite( o.data(), 8, o.size(), f );
    std::fwrite( nbrs.data(), 8, nbrs.size(), f );
    std::fwrite( ref.data(), 8, ref.size(), f );
    std::fclose( f );
    std::fprintf( stderr, "dump_csr: %s  dim %d  n %lld  neighbors %lld\n",
                  path.c_str(), dim, ( long long ) n, ( long long ) nv );
    return true;
}
