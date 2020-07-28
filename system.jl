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
    rc = find_vortex(u)
    @debug "Angular momentum limit" Jmax
    
    a = log(10,residual)
    g_tols = 10 .^ (-2:-0.5:a)
    if isempty(g_tols) || g_tols[end] ≉ residual
        push!(g_tols, residual)
    end
    
    mintol = Inf
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
                j = 1
            elseif abs(find_vortex(u)-rc) < h
                j = 2
            elseif gtol < mintol
                mintol = gtol
                @debug "Orbit" gtol J=Ju Ω Ωtol=(Ωs[2]-Ωs[1])/2
                continue
            else
                continue
            end
            @debug ((j==1) ? "Free" : "Central") J=Ju Ω Ωtol=(Ωs[2]-Ωs[1])/2
            # TODO 
            (Ωs[j] == Ω || Ωs[1] ≥ Ωs[2]) && return NaN, u
            Ωs[j] = Ω
            break
        end
    
        0.1Jmax < real(dot(u, J(u))) && abs(find_vortex(u)-rc) ≥ h &&  break
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

function find_vortex(u)
    P, Q = poles(u)
    ixs = abs.(P) .> 0.5maximum(abs.(P))
    regress_core(u, ixs)
end

function find_moat(u)
    P, Q = poles(u)
    v = @. (R-w/2 < r < R+w/2)*abs(P+conj(Q))/abs(u)
    ixs = v .> 0.5maximum(v)
    regress_core(u, ixs)
end

function regress_core(u, ixs)
    ixs = expand_box(box(ixs))[:]
    a, b, c = [z[ixs] conj(z[ixs]) ones(size(z[ixs]))] \ u[ixs]
    (b*conj(c)-conj(a)*c)/(abs2(a)-abs2(b))
end

function expand_box(bb::CartesianIndices{2})
    e1, e2 = ((1,0), (0,1)) .|> CartesianIndex
    # TODO raise issue on CartesianIndex half-pregnancy
    if length(bb) ==1
        bb = bb[]
        bb = [
            bb-e1-e2 bb-e1 bb-e1+e2;
            bb-e2 bb bb+e2;
            bb+e1-e2 bb+e1 bb+e1+e2] 
    elseif size(bb,1) ==1
        bb = [bb .- e1; bb; bb .+ e1] 
    elseif size(bb,2) ==1
        bb = [bb .- e2 bb bb .+ e2]
    end
    bb
end

"Bounding box of cartesian indices"
function box(cixs::AbstractVector{CartesianIndex{2}})
   j1 = k1 = typemax(Int)
   j2 = k2 = typemin(Int)
   for c in cixs
       j1 = min(j1,c[1])
       j2 = max(j2,c[1])
       k1 = min(k1,c[2])
       k2 = max(k2,c[2])
   end
   CartesianIndices((j1:j2, k1:k2))
end

box(ixs::AbstractMatrix{Bool}) = box(keys(ixs)[ixs])

function slice(u)
    j = N÷2
    sum(u[j:j+1,:], dims=1)[:]/2
end

cscl(z) = (abs(z)>0) ? z/abs(z)*log(10,abs(z)) : Complex(1e-10)

"Imprint an exp(iθ) phase while keeping the density"
function imprint_phase(u)
    r₀ = find_vortex(u)
    @. abs(u)*(z-r₀)/abs(z-r₀)
end

berry_diff(u,v) = imag(sum(conj(v).*u))