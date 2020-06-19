# BdG for an offset vortex, with numerical improvements from ../basics

using LinearAlgebra, BandedMatrices, Arpack, Optim, DifferentialEquations
using Plots, ComplexPhasePortrait, Printf

C = 3000
N = 60
Ω = 0.28
dts = 10 .^ (-5:-0.5:-7.5)	# residual
ats = 10 .^ (-2:-0.25:-4)	# time step
sfile = "orb.jld2"

r₀ = 1.9		# offset of imprinted phase

h = sqrt(√2*π/N)
y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op(Float64[-1/2, 0, 1/2])
∂² = (1/h^2).*op(Float64[1, -2, 1])

# Minimise the energy 
#
# E(ψ) = -∫ψ*∇²ψ/2 + V|ψ|²+g/2·|ψ|⁴
#
# The GPE functional L(ψ) is the gradient required by Optim.

T(ψ) = -(∂²*ψ+ψ*∂²')/2
U(ψ) = C/h*abs2.(ψ)
J(ψ) = -1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
L(ψ) = T(ψ)+(V+U(ψ)).*ψ+J(ψ)
K(ψ) = T(ψ)+(V+U(ψ)).*ψ		# lab frame
H(ψ) = T(ψ)+(V+U(ψ)/2).*ψ+J(ψ)
E(xy) = sum(conj.(togrid(xy)).*H(togrid(xy))) |> real
grdt!(buf,xy) = copyto!(buf, 2*L(togrid(xy))[:])
togrid(xy) = reshape(xy, size(z))

# starting point for relaxation
φ = @. cos(π*x/(N+1)/h)*cos(π*y/(N+1)/h) |> Complex
φ .*= (z.-r₀)
φ ./= norm(φ)

dSs = []
dsteps = Float64[]
"Starting the real work" |> println
flush(stdout)
for r = dts
    result = optimize(E, grdt!, φ[:],
         ConjugateGradient(manifold=Sphere()),
         Optim.Options(iterations=10_000, g_tol=r, allow_f_increases=true)
    );
    φ .= result.minimizer |> togrid
    push!(dsteps, result.iterations)
    "Relaxed to residual $(r) in $(result.iterations) steps" |> print
    flush(stdout)
    
    P = ODEProblem((ψ,_,_)->-1im*K(ψ), φ, (0.0,0.75))
    S = solve(P, RK4(), adaptive=false, dt=minimum(ats), saveat=0.05)
    push!(dSs, S)
    " and solved dynamics" |> println
    flush(stdout)
end
dsteps = cumsum(dsteps)
P = ODEProblem((ψ,_,_)->-1im*K(ψ), φ, (0.0,0.75))

aSs = []
for a = ats
    S = solve(P, RK4(), adaptive=false, dt=a, saveat=0.05)
    push!(aSs, S)
    "Solved dynamics with time step $(a)" |> println
    flush(stdout)
end

struct GPEMatrix <: AbstractMatrix{Complex{Float64}}
    ψ::Matrix{Complex{Float64}}
end

Base.size(A::GPEMatrix) = (N^2, N^2)

function LinearAlgebra.mul!(uout::AbstractVector, A::GPEMatrix, uin::AbstractVector)
    u = reshape(uin, N, N) 
    Au = reshape(uout, N, N)
    Au .= T(u)+(V+U(A.ψ)).*u+J(u)
   uout
end

# ew,ev = eigs(GPEMatrix(q); nev=16, which=:SR)

function er(u, ev)
    j = argmax(abs.(ev'*u[:]))
    u0 = ev[:,j] |> togrid
    u - dot(u0,u)*u0
end

function scstrm(u)
    # expand u over self-consistent eigenstates in rotating frame
    ews,evs = eigs(GPEMatrix(q); nev=30, which=:SR)
    ewl,evl = eigs(GPEMatrix(q); nev=30, which=:LR)
    ew = [ews; ewl]
    ev = [evs evl]
    @assert maximum(abs.(imag.(ew))) < 1e-10
    ew = real.(ew)
    ew, ev
end

function labplot!(P, u, ew, ev, clr=:black)
    cs = abs.(ev'*u[:])
    ixs = cs .> 1e-20
    scatter!(P, ew[ixs], cs[ixs], mc=clr, msw=0, ms=3, yscale=:log10, leg=:none)
end

function diagplot(S, j)
    P1 = plot()
    P3 = plot()
    P2 = nothing
    for k = 1:length(S)
        ew, ev = scstrm(S[j])
        e = er(S[k], ev)
        scatter!(P1, S.t[k:k], [norm(e)],
            ms = 3, mc=:black, msw=0, leg=:none)
        if k == j
            scatter!(P1, S.t[k:k], [norm(e)],
                ms = 4, mc=:red, msw=0, leg=:none)
            P2 = zplot(e)
            labplot!(P3, S[k], ew, ev, :red)
        elseif k == 1
            labplot!(P3, S[j], ew, ev, :black)
        end
    end
    plot(P1, P2, P3, layout=@layout [a b; c])
end

zplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
argplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)), aspect_ratio=1)
argplot(ψ::Matrix{<:Real}) = argplot(Complex.(ψ))
