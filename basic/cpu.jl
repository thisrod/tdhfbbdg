# Test run to validate SHO

using LinearAlgebra, BandedMatrices, Optim, DifferentialEquations, JLD2
using Plots, ComplexPhasePortrait

Ea = √2

# C = 10.0
C = 10_000.0

N = 60
h = sqrt(√2*π/N)

# remember SHO, E = T/2 + V/2
Tmax = π^2/h^2
Vmax = N^2*h^2/2

    y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
    r = abs.(z)
    V = r² = abs2.(z)
    
    # Exact solution on infinite domain
    
    φa = @. exp(-r²/√2)
    φa ./= norm(φa)
    
    # starting point for gradient descent
    φ = @. cos(π*x/(N+1)/h)*cos(π*y/(N+1)/h)
    φ ./= norm(φ)
    
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
    
T(ψ) = -(∂²*ψ+ψ*∂²')/2
U(ψ) = C/h*abs2.(ψ)
L(ψ) = T(ψ)+(V+U(ψ)).*ψ
H(ψ) = T(ψ)+(V+U(ψ)/2).*ψ
E(xy) = sum(conj.(togrid(xy)).*H(togrid(xy))) |> real
grdt!(buf,xy) = copyto!(buf, 2*L(togrid(xy))[:])
togrid(xy) = reshape(xy, size(z))
    
    result = optimize(E, grdt!, Complex.(φ)[:],
         ConjugateGradient(manifold=Sphere()),
         Optim.Options(iterations=10_000, g_tol=1e-10, allow_f_increases=true)
    )
    init = togrid(result.minimizer)
    
    P = ODEProblem((ψ,_,_)->-1im*L(ψ), init, (0.0,1.0))

# scatter(ats,aers, xscale=:log10, yscale=:log10, leg=:none)

function crds(u)
    N = size(u,1)
    h = 0.6/√N
    h/2*(1-N:2:N-1)
end
   
zplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
argplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)), aspect_ratio=1)
argplot(ψ::Matrix{<:Real}) = argplot(Complex.(ψ))


u = similar(z)
Hmat = similar(z, N^2, N^2)
for j = 1:N^2
    u .= 0
    u[j] = 1
    Hmat[:,j] = (T(u)+(V+U(init)).*u)[:]
end

Hmat = real.(Hmat)
ew, ev = eigen(Hmat)

ψ₀ = togrid(ev[:,1])
er(u) = u - dot(ψ₀, u)*ψ₀

cpts = ev'*φa[:]

tix(S,t) = argmin(abs.(S.t .- t))
    
ats = 10 .^ (-8:-0.5:-12)
rts = 10 .^ (-8:-0.5:-12)
aers = Float64[]
rers = Float64[]
for a = ats
    T = solve(P, RK4(), abstol=a, reltol=minimum(rts))
    q = T[tix(T,0.75)]
    push!(aers, q - dot(ψ₀, q)*ψ₀ |> norm)
end
for r = rts
    T = solve(P, RK4(), abstol=minimum(ats), reltol=r)
    q = T[tix(T,0.75)]
    push!(rers, q - dot(ψ₀, q)*ψ₀ |> norm)
end
S = solve(P, RK4(), abstol=minimum(ats), reltol=minimum(rts))
tix(t) = tix(S,t)

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
