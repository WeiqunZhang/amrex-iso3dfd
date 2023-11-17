#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_ParmParse.H>
#include <iostream>

using namespace amrex;

static constexpr float dt = 0.002f;
static constexpr float dxyz = 50.0f;
static constexpr int kHalfLength = 8;

namespace {
    int use_array4 = false;
    int use_array4_hack = false;
}

void Initialize (FArrayBox& prev, FArrayBox& next, FArrayBox& vel)
{
    std::cout << "Initializing ... \n";

    prev.template setVal<RunOn::Device>(0.0f);
    next.template setVal<RunOn::Device>(0.0f);
    vel.template setVal<RunOn::Device>(2250000.0f * dt * dt);

    // Add a source to initial wavefield as an initial condition
    float val = 1.f;
    auto nx = vel.box().length(0);
    auto ny = vel.box().length(1);
    auto nz = vel.box().length(2);
    for (int s = 5; s >= 0; s--) {
        Box b(IntVect(nx/4-s  , ny/4-s  , nz/2-s  ),
              IntVect(nx/4+s-1, ny/4+s+1, nz/2+s-1));
        vel.template setVal<RunOn::Device>(val, b);
        val *= 10.f;
    }
}

void Iso3dfd (FArrayBox& nextfab, FArrayBox& prevfab, FArrayBox const& velfab,
              Gpu::DeviceVector<float> const& coeffdv, int num_iterations)
{
    Box const& b = amrex::grow(nextfab.box(), -kHalfLength);
    auto const* coeff = coeffdv.data();
    for (int it = 0; it < num_iterations; ++it) {
        auto const& next = (it % 2 == 0) ? nextfab.array() : prevfab.array();
        auto const& prev = (it % 2 == 0) ? prevfab.const_array() : nextfab.const_array();
        auto const& vel = velfab.const_array();
        if (use_array4) {
            if (use_array4_hack) {
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    auto *pn = next.ptr(i,j,k);
                    auto const* pp = prev.ptr(i,j,k);
                    auto const* pv = vel.ptr(i,j,k);
                    float value = (*pp) * coeff[0];
#pragma unroll(kHalfLength)
                    for (int ir = 1; ir <= kHalfLength; ++ir) {
                        value += coeff[ir] * (pp[ ir] +
                                              pp[-ir] +
                                              pp[ ir*prev.jstride] +
                                              pp[-ir*prev.jstride] +
                                              pp[ ir*prev.kstride] +
                                              pp[-ir*prev.kstride]);
                    }
                    *pn = 2.0f * (*pp) - (*pn) + value*(*pv);
                });
            } else {
                ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    float value = prev(i,j,k) * coeff[0];
#pragma unroll(kHalfLength)
                    for (int ir = 1; ir <= kHalfLength; ++ir) {
                        value += coeff[ir] * (prev(i-ir,j   ,k   ) +
                                              prev(i+ir,j   ,k   ) +
                                              prev(i   ,j-ir,k   ) +
                                              prev(i   ,j+ir,k   ) +
                                              prev(i   ,j   ,k-ir) +
                                              prev(i   ,j   ,k+ir));
                    }
                    next(i,j,k) = 2.0f * prev(i,j,k) - next(i,j,k) + value*vel(i,j,k);
                });
            }
        } else {
            auto* pn = next.ptr(0,0,0);
            auto const* pp = prev.ptr(0,0,0);
            auto const* pv = vel.ptr(0,0,0);
            auto jstride = next.jstride;
            auto kstride = next.kstride;
            ParallelFor(b, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                auto offset = i + j*jstride + k*kstride;
                float value = pp[offset] * coeff[0];
#pragma unroll(kHalfLength)
                for (int ir = 1; ir <= kHalfLength; ++ir) {
                    value += coeff[ir] * (pp[offset+ir] +
                                          pp[offset-ir] +
                                          pp[offset+ir*jstride] +
                                          pp[offset-ir*jstride] +
                                          pp[offset+ir*kstride] +
                                          pp[offset-ir*kstride]);
                }
                pn[offset] = 2.0f * pp[offset] - pn[offset] + value*pv[offset];
            });
        }
    }
}

void PrintStats (double time, Box const& domain, int nIterations)
{
    auto normalized_time = time / double(nIterations);
    auto throughput_mpoints = domain.d_numPts() / normalized_time / 1.e6;

    auto mflops = (7.0 * double(kHalfLength) + 5.0) * throughput_mpoints / 1.e3;
    auto mbytes = 12.0 * throughput_mpoints / 1.e3;

    std::cout << "--------------------------------------\n";
    std::cout << "time         : " << time << " secs\n";
    std::cout << "throughput   : " << throughput_mpoints << " Mpts/s\n";
    std::cout << "flops        : " << mflops << " GFlops\n";
    std::cout << "bytes        : " << mbytes << " GBytes/s\n";
    std::cout << "\n--------------------------------------\n";
    std::cout << "\n--------------------------------------\n";

}

void main_main ()
{
    static_assert(std::is_same_v<float,Real>);

    std::array<int,3> grid_sizes{256,256,256};
    int num_iterations = 10;
    {
        ParmParse pp;
        pp.query("grid_sizes", grid_sizes);
        pp.query("iterations", num_iterations);
        pp.query("use_array4", use_array4);
        pp.query("use_array4_hack", use_array4_hack);
    }

    Box domain(IntVect(0),IntVect(grid_sizes[0]-1,
                                  grid_sizes[1]-1,
                                  grid_sizes[2]-1));
    FArrayBox prev, next, vel;
    {
        Box fabbox = amrex::grow(domain,kHalfLength);
        prev.resize(fabbox,1);
        next.resize(fabbox,1);
        vel.resize(fabbox,1);
    }

    // Compute coefficients to be used in wavefield update
    Array<float,kHalfLength+1> coeff
        {-3.0548446f,   +1.7777778f,     -3.1111111e-1f,
         +7.572087e-2f, -1.76767677e-2f, +3.480962e-3f,
         -5.180005e-4f, +5.074287e-5f,   -2.42812e-6f};
    // Apply the DX DY and DZ to coefficients
    coeff[0] = (3.0f * coeff[0]) / (dxyz * dxyz);
    for (int i = 1; i <= kHalfLength; i++) {
        coeff[i] = coeff[i] / (dxyz * dxyz);
    }
    Gpu::DeviceVector<float> coeff_dv(coeff.size());
    Gpu::copyAsync(Gpu::hostToDevice, coeff.begin(), coeff.end(), coeff_dv.begin());

    std::cout << "Grid Sizes: " << grid_sizes[0] << " " << grid_sizes[1] << " "
              << grid_sizes[2] << "\n";
    std::cout << "Memory Usage: " << ((3*prev.nBytes()) / (1024 * 1024)) << " MB\n";

    Initialize(prev, next, vel);
    Gpu::streamSynchronize();

    Iso3dfd(next, prev, vel, coeff_dv, 2); // warm up
    Gpu::streamSynchronize();

    auto t0 = amrex::second();
    Iso3dfd(next, prev, vel, coeff_dv, num_iterations);
    Gpu::streamSynchronize();
    auto t1 = amrex::second();

    PrintStats(t1-t0, domain, num_iterations);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        amrex::Print() << "\n\n";
        main_main();
        amrex::Print() << "\n\n";
    }
    amrex::Finalize();
}
