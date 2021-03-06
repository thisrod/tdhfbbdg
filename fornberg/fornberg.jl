# Julia port of Program 28 from SMM

# TODO
# find SHO ground state by Optim
# add repulsion
# find offset vortex
# debug fit instability for large N
# derive Clenshaw-Curtis weights by fourier transforms
# attempt to derive Clenshaw-Curtis weights in closed form
# Barycentric interpolation

using LinearAlgebra, ToeplitzMatrices, Polynomials, Optim
using Statistics: mean
using Plots, ComplexPhasePortrait

R = 2.0		# disk domain radius
Ω = 0.1
C = 50

# Chebyshev grid and derivative matrix on [-1,1]
function cheb(N::Integer)
    N == 0 && return (0, 1)
    x = cos.(π*(0:N)/N)
    c = [2; ones(N-1,1); 2].*(-1).^(0:N)
    dX = x.-x'
    D  = c ./ c' ./ (dX+I)					# off-diagonal entries
    D  -= sum(D; dims=2)[:] |> Diagonal		# diagonal entries
    D, x
end

# radial derivatives
N = 25;  @assert N % 2 == 1
N2 = (N-1)÷2
D, r = cheb(N)
r *= R
D /= R
D2 = D^2
D1 = D2[2:N2+1,2:N2+1]; D2 = D2[2:N2+1,N:-1:N2+2]
E1 =  D[2:N2+1,2:N2+1]; E2 =  D[2:N2+1,N:-1:N2+2]

# circular derivatives
M = 20;   @assert M % 2 == 0
dθ = 2π/M;  θ = dθ*(1:M); M2 = M÷2
rc = [0; 0.5*(-1).^(1:M-1).*cot.(dθ*(1:M-1)/2)]
Dθ = Toeplitz(rc, -rc)
rc = [-π^2/(3*dθ^2)-1/6;  0.5*(-1).^(2:M)./sin.(dθ*(1:M-1)/2).^2]
D2θ = Toeplitz(rc, rc)

# Laplacian in polar coordinates:
rint = r[2:N2+1]
S = 1 ./ rint |> Diagonal
Z = zeros(M2,M2)
L = kron(D1+S*E1,Matrix(I,M,M)) +
    kron(D2+S*E2,[Z I;I Z]) +
    kron(S^2,D2θ)

# Quadrature weights
# Try Julia's Tsebyshev polynomials as well
# For stable interpolation, wrap sinc around a cylinder

w = zeros(1,N2)
for j = 1:N2
    v = zero(r)
    v[j+1] = v[end-j] = 1
    p = Polynomial([0,1])*fit(r,v)
    p = integrate(p)
    w[j] = 2π*(p(R)-p(0))/M
end

# unweight is L² normalized if x is l² normalized
# x is a splatted vector
function unweight(x)
    x = reshape(x, M, N2)
    y = x./sqrt.(w)
    y[:]
end

V = kron(Diagonal(rint.^2),Matrix(I,M,M))
J = 1im*kron(Matrix(I,N2,N2), Dθ)
H = -L/2 + V - Ω*J

# eigenmodes
index = 1:10
ωs, ev = eigen(H)
ii = sortperm(ωs; by=abs)[index]
ev = ev[:,ii]

# Optim ground state
# The vector x is multiplied by the quadature weights, so that a
# normalised x represents a normalised wave function.  The function
# unweight inverts this.

f(x) = dot(x, H*x) |> real
g(x) = 2*H*x		# linear unweight is its own derivative
g!(stor,x) = copyto!(stor,g(x))
init = repeat(rint'.*exp.(-rint.^2)', M, 1)
init = Complex.(init[:])
result = Optim.optimize(f, g!, init,
    Optim.ConjugateGradient(manifold=Optim.Sphere()),
    Optim.Options(iterations = 10_000, allow_f_increases=true))

mode(j) = reshape(ev[:,j], M, N2)

yy = -R:R/50:R;  xx = yy'
M2 = M÷2
function interpolate(u)
    rr = hypot.(xx,yy)
    hh = atan.(yy,xx)
    uu = zeros(eltype(u), size(rr))
    for j = 1:M2
        v = [0; u[j,:]; reverse(u[j+M2,:]); 0];
        p = fit(r,v)
        uu .+= (rr .≤ R).*p.(rr).*sinc.(hh.-θ[j])
        uu .+= (rr .≤ R).*p.(-rr).*sinc.(hh.-θ[j+M2])
    end
    uu
end

sinc(x) = (x ≈ 0) ? zero(x) : sin(π*x/dθ)/(2π/dθ)/tan(x/2)

zplot(ψ) = plot(xx[:], yy, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
argplot(ψ) = plot(xx[:], yy, portrait(reverse(ψ,dims=1)), aspect_ratio=1)
argplot(ψ::Matrix{<:Real}) = argplot(Complex.(ψ))

# Plots
rr = (rint[1:end-1] .+ rint[2:end])/2
rr = [1; rr; 0]
hh = (θ[1:end-1] .+ θ[2:end])/2
hh = [dθ/2;hh;θ[end]+dθ/2]

plr((r, h)) = (r*cos(h), r*sin(h))
rh(j,k) = [(rr[j], hh[k]), (rr[j+1], hh[k]), (rr[j+1], hh[k+1]), (rr[j], hh[k+1]), (rr[j], hh[k])]
vtxs(j,k) = plr.(rh(j,k))
ppix(j,k) = vtxs(j,k) |> Shape

function rplot(u)
    P = plot(
        bg = :black,
        xlim = (-R, R),
        ylim = (-R, R),
        framestyle = :none,
        legend = false,
        aspect_ratio = 1
    )
    clrs = portrait(u).*abs2.(u)/maximum(abs2.(u))
    for j = 1:length(rr)-1
        for k = 1:length(hh)-1
            plot!(P, ppix(j,k), fillcolor = clrs[k,j], lw=0, leg=:none)
        end
    end
    display(P)
end

mode(j) = reshape(ev[:,j], M, N2)