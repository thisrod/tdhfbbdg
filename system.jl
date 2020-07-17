# Given C, N, l, set up rotating trap system

using LinearAlgebra, BandedMatrices, Optim
using Statistics: mean

h = min(l/(N+1), sqrt(√2*π/N))
y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
r = abs.(z)
V = r² = abs2.(z)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op(Float64[-1/2, 0, 1/2])
∂² = (1/h^2).*op(Float64[1, -2, 1])

T(ψ) = -(∂²*ψ+ψ*∂²')/2
U(ψ) = C/h*abs2.(ψ)
J(ψ) = -1im*(x.*(∂*ψ)-y.*(ψ*∂'))
L(ψ,Ω) = L(ψ)-Ω*J(ψ)
L(ψ) = T(ψ)+(V+U(ψ)).*ψ
H(ψ,Ω) = H(ψ)-Ω*J(ψ)
H(ψ) = T(ψ)+(V+U(ψ)/2).*ψ

# Vortex-free cloud satisfying zero BC

φ = @.cos(π*x/(N+1)/h)*cos(π*y/(N+1)/h) |> Complex
φ ./= norm(φ)

function ground_state(initial, Ω, g_tol)
    result = optimize(
        ψ -> dot(ψ,H(ψ,Ω)) |> real,
        (buf,ψ) -> copyto!(buf, 2*L(ψ,Ω)),
        initial,
        ConjugateGradient(manifold=Sphere()),
        Optim.Options(iterations=10_000, g_tol=g_tol, allow_f_increases=true)
    )
    Optim.converged(result) || error("Ground state failed to converge")
    result.minimizer
end

function op2mat(f)
    u = similar(z)
    M = similar(z,N^2,N^2)
    for j = eachindex(u)
        u .= 0
        u[j] = 1
        M[:,j] = f(u)[:]
    end
    M
end

# Wrapper for Arpack

struct Operator <: AbstractMatrix{Complex{Float64}}
    f
end

Base.size(H::Operator) = (N^2, N^2)
function LinearAlgebra.mul!(y, H::Operator, x)
    y .= H.f(reshape(x,N,N))[:]
end

# Adjust Ω to acquire an orbiting vortex from an imprint at r₀
# TODO think about Ju tolerance for extreme orbits
function orbit_frequency(r₀, residual; Ωs = [0.0, 0.6], moat=false)
    # find J limit in case of moat
    u = z.*φ;
    u ./= norm(u);
    u .= ground_state(u, 0.0, residual);
    Jmax = dot(u, J(u)) |> real
    @info "Angular momentum limit" Jmax
    
    a = log(10,residual)
    g_tols = 10 .^ (-2:-0.5:a)
    g_tols[end] ≈ residual || push!(g_tols, residual)
    
    
    while true
        Ω = mean(Ωs)
        u .= φ;
        @. u *= (z-r₀);
        if moat
            @. u *= conj(z+R)
        end
        u ./= norm(u);
        
       for gtol = g_tols
            u .= ground_state(u, Ω, gtol);
            Ju = dot(u, J(u)) |> real
            if Ju < 0.1Jmax
                @debug "Free" gtol Ω r₀ Ju
                j = 1
            elseif Ju > 0.9Jmax
                @debug "Central" gtol Ω r₀ Ju
                j = 2
            else
                # TODO only show the lowest gtol in a sequence
                @debug "Orbit" gtol Ω r₀ Ju
                continue
            end
            @info "Rotation convergence" Ω Ωtol=(Ωs[2]-Ωs[1])/2
            (Ωs[j] == Ω || Ωs[1] ≥ Ωs[2]) && return NaN, u
            Ωs[j] = Ω
            break
        end
    
        0.1 < real(dot(u, J(u))) < 0.9Jmax  &&  break
    end
    
    mean(Ωs), u
end

# acquire a vortex orbiting at r₀
function acquire_orbit(r₀, residual, tol=h)
    Ω, q = orbit_frequency(r₀, residual);
    rv = abs(find_vortex(q))
    a = r₀ - rv
    @assert a > 0
    # Intialise rs to [inside, outside]
    rs = [NaN, NaN]
    for j = 0:4
        r₁ = r₀
        rv = 0.0
        while rv < r₀
            rs[1] = r₁
            r₁ += a/2^j
            Ω, q = orbit_frequency(r₁, residual);
            isnan(Ω) && break
            rv = abs(find_vortex(q))
        end
        if isnan(Ω)
            continue
        elseif rv ≥ r₀
            rs[2] = r₁
            break
        end
    end
    any(isnan, rs) && return NaN, NaN, q
    
    # bisect
    while abs(rv-r₀) > tol
        r₁ = mean(rs)
        Ω, q = orbit_frequency(r₁, residual);
        rv = abs(find_vortex(q))
        if rv < r₀
            @debug "Inside" r₁ rv Ω
            j = 1
        else
            @debug "outside" r₁ rv Ω
            j = 2
        end
        @info "Imprint convergence" r=r₁ rtol=(rs[2]-rs[1])/2
        (rs[j] == r₁ || rs[1] ≥ rs[2]) && return NaN, NaN, q
        rs[j] = r₁
    end
    
    rv, Ω, q
end

"""
    P, Q = poles(u)

Return the Wirtinger derivatives of u
"""
function poles(u)
    u = complex.(u)
    rs = (-1:1)' .+ 1im*(-1:1)
    rs /= h*sum(abs2.(rs))
    conv(u, rs) = [rs.*u[j:j+2,k:k+2] |> sum for j = 1:N-2, k = 1:N-2]
    P = zero(u)
    P[2:end-1,2:end-1] .= conv(u, conj(rs))
    Q = zero(u)
    Q[2:end-1,2:end-1] .= conv(u, rs)
    P, Q
end

find_vortex(u) = find_vortex(u, Inf)
function find_vortex(u, R)
    w = poles(u) |> first .|> abs
    @. w *= abs(z) < R
    z[argmax(w)]
end

function slice(u)
    j = N÷2
    sum(u[j:j+1,:], dims=1)[:]/2
end
