# brute force zernike polynomials

using LinearAlgebra, Plots, Optim, Arpack, Jacobi

using Revise
using Superfluids

default(:legend, :none)

function zpoly(z,n,m)
    (abs(m) > n || isodd(n-abs(m)) || abs(z)>1) && return zero(z)
    rpoly(abs(z), n, m)*(z/abs(z))^m
end

function rpoly(r,n,m)
    r^abs(m)*jacobi(2r^2-1, (n-abs(m))÷2, 0, abs(m))
end

function fourier(m)
    normalize(@. (abs(z) < d.h*d.n/2)*(z/abs(z))^m)
end

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
d = FDDiscretisation{2}(66, 0.3)
g_tol = 1e-7

L, H, J = Superfluids.operators(s,d,:L,:H,:J)

ψ = steady_state(s,d)
μ = dot(L(ψ), ψ) |> real
E₀ = dot(H(ψ), ψ) |> real
R_TF = sqrt(2μ)

Ω = 0.3

# Optimizing E gives better answer than optimizing residual
result = optimize(0.2R_TF, 0.5R_TF, abs_tol=g_tol) do r
    q = steady_state(s, d; rvs=Complex{Float64}[-r, r], Ω, g_tol, iterations=1000)
    real(dot(H(q;Ω),q))
end
r = result.minimizer

q = steady_state(s, d; rvs=Complex{Float64}[-r, r], Ω, g_tol, iterations=1000)

# ws, us, vs = bdg_modes(s, d, q, Ω, 10, nev=100)

B = Superfluids.BdGmatrix(s,d,Ω,q)
ew, ev = eigs(B, nev=20, which=:SM)
ws, us, vs = Superfluids.bdg_output(d,ew,ev)

l = d.n*d.h/2
z = Superfluids.argand(d)

ixs = [(n,m) for n = 0:50 for m = -n:2:n]

u = normalize(us[1])
cs = [dot(normalize(zpoly.(z/l,n,m)), u) for (n, m) in ixs]

# scatter([ix[2] for ix in ixs], abs2.(cs), yscale=:log10, ms=2, msw=0, mc=:black, xlabel="m")

ds = [dot(normalize(zpoly.(z/l,n,m)), u)
    for n = 0:2:100 for m = -2:2:6]
filter!(d->!isnan(d), ds)

ips = Complex{Float64}[]
nmax = 6
for n = 0:nmax, m = -n:2:n, nn = 0:nmax, mm = -nn:2:nn
    push!(ips, dot(normalize(zpoly.(z/l,n,m)), normalize(zpoly.(z/l,nn,mm))))
end
sz = (nmax+1)*(nmax+2)÷2
ips = reshape(ips, sz,sz)

# use pairplots.jl
# scatter(-10:10, [abs2(dot(fourier(m),u)) for m = -10:10], yscale=:log10)