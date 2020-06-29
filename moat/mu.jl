# Moat potential

using LinearAlgebra, BandedMatrices, Optim, DifferentialEquations, JLD2
using Plots, ComplexPhasePortrait

Ea = √2

# C = 1.0
# N = 60
# dts = 10 .^ (-8:-0.5:-16)	# residual
# ats = 10 .^ (-2:-0.25:-3.5)	# time step
# sfile = "cvg_0.jld2"

# C = 10.0
# N = 60
# dts = 10 .^ (-8:-0.5:-15)	# residual
# ats = 10 .^ (-2:-0.25:-3.75)	# time step
# sfile = "cvg_1.jld2"

# C = 100.0
# N = 60
# dts = 10 .^ (-8:-0.5:-14)	# residual
# ats = 10 .^ (-2:-0.25:-4)	# time step
# sfile = "cvg_2.jld2"

C = 10_000.0
N = 100
dts = 10 ^ -11.5	# residual
ats = 10 ^ -4.5	# time step

μoff = 4
R = 1.9
w = 0.2

h = sqrt(√2*π/N)

# remember SHO, E = T/2 + V/2
Tmax = π^2/h^2
Vmax = N^2*h^2/2

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
r = abs.(z)
V = r² = abs2.(z)
@. V += 100*exp(-(r-R)^2/2/w^2)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end

# Hard zero boundary conditions.
∂² = (1/h^2).*op(Float64[1, -2, 1])

# Minimise the energy 
#
# E(ψ) = -∫ψ*∇²ψ/2 + V|ψ|²
#
# The GPE functional L(ψ) is the gradient required by Optim.

μL = 0.0
T(ψ) = -(∂²*ψ+ψ*∂²')/2
U(ψ) = C/h*abs2.(ψ)
L(ψ) = T(ψ)+(V+U(ψ).-μL).*ψ
H(ψ) = T(ψ)+(V+U(ψ)/2).*ψ
E(xy) = sum(conj.(togrid(xy)).*H(togrid(xy))) |> real
grdt!(buf,xy) = copyto!(buf, 2*L(togrid(xy))[:])
togrid(xy) = reshape(xy, size(z))
    
tix(S,t) = argmin(abs.(S.t .- t))

# starting point for relaxation
φ = @. cos(π*x/(N+1)/h)*cos(π*y/(N+1)/h) |> Complex
φ ./= norm(φ)

result = optimize(E, grdt!, φ[:],
     ConjugateGradient(manifold=Sphere()),
     Optim.Options(iterations=10_000, g_tol=dts[], allow_f_increases=true)
);
φ .= result.minimizer |> togrid

# Set chemical potential to zero outside the moat
μL = dot(φ, L(φ)) |> real
P = ODEProblem((ψ,_,_)->-1im*L(ψ), φ, (0.0,5.0))

# effective Hamiltonian including ground state repulsion

# u = similar(z)
# Hmat = similar(z, N^2, N^2)
# for j = 1:N^2
#     u .= 0
#     u[j] = 1
#     Hmat[:,j] = (T(u)+(V+U(φ)).*u)[:]
# end
# Hmat = real.(Hmat)
# ew, ev = eigen(Hmat)
# ψ₀ = togrid(ev[:,1])

er(u) = u - dot(ψ₀, u)*ψ₀

tix(t) = tix(S,t)

t(x) = (tanh(x)+1)/2
@. V += μoff*t((R+r)/w)*t((R-r)/w)
S = solve(P, RK4(), adaptive=false, dt=minimum(ats), saveat=0.5)

function labplot(u)
    cs = abs.(ev'*u[:])
    ixs = cs .> 1e-20
    ixs[1] = false
    scatter(ew[ixs], cs[ixs], mc=:black, msw=0, ms=3, yscale=:log10, leg=:none)
end

function labplot!(u)
    cs = abs.(ev'*u[:])
    ixs = cs .> 1e-20
    ixs[1] = false
    scatter!(ew[ixs], cs[ixs], mc=:red, msw=0, ms=3, yscale=:log10, leg=:none)
end

function diagplot(j)
    P1 = scatter(S.t, [norm(er(S[j])) for j = eachindex(S)],
        ms = 3, mc=:black, msw=0, leg=:none)
    scatter!(S.t, [abs(dot(ψ₀,S[j])) - 1 for j = eachindex(S)],
        ms = 3, mc=:blue, msw=0, leg=:none)
    scatter!(S.t[j:j], [norm(er(S[j]))],
        ms = 4, mc=:red, msw=0, leg=:none)
    P2 = zplot(er(S[j]))
    P3 = labplot(S[j])
    labplot!(S[1])
    plot(P1, P2, P3, layout=@layout [a b; c])
end
   
zplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
argplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)), aspect_ratio=1)
argplot(ψ::Matrix{<:Real}) = argplot(Complex.(ψ))


function convplot()
    scatter(asteps,aers, xscale=:log10, yscale=:log10,
        label="time step", leg=:bottomleft)
    scatter!(dsteps,ders, label="residual")
    scatter!([1.0], ders[end:end], mc=:black, label="best")
    xlabel!("relative work")
    ylabel!("error component")
    title!("Convergence at t = 0.75")
end

# T0 = dot(ψ₀, T(ψ₀))
# U0 = dot(ψ₀, U(ψ₀).*ψ₀)
# V0 = dot(ψ₀, V.*ψ₀)

function slice(u)
    j = N÷2
    sum(u[j:j+1,:], dims=1)[:]/2
end

# scatter(y, slice(W), msw=0, mc=:black, label="V")
# scatter!(y, 750*slice(real.(φ)).+E0, msw=0, mc=:gray, label="q")