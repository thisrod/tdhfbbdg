# Given C, N, l, set up rotating trap system

using LinearAlgebra, BandedMatrices, Optim

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

# Acquire a vortex orbiting at radius r₀
function acquire_orbit(r₀, residual=1e-6)
    # find central vortex for J limit with moat
    
    u = z.*copy(φ)
    u ./= norm(u)
    u .= ground_state(u, 0.0, residual)
    Jmax = dot(u, J(u)) |> real
    
    # invariant: when (Ωs[1,j], rs[1,j]) is relaxed to gtol[j], it
    # gives a vortex inside the target radius.  [2,j] is outside,
    # where vortex-free counts as ∞.
    
    g_tols = 10 .^ (-3:-0.5:log(10,residual))
    Ωs = repeat([0.0, 0.6], 1, length(g_tols))
    rs = repeat([r₀, r₀], 1, length(g_tols))
    
    while true
        Ω = mean(Ωs)
        r₁ = rv = mean(rs)
        u .= φ;
        @. u *= (z-r₁);
        u ./= norm(u);
        
#        for gtol = g_tols
j = 1
gtol = g_tols[j]
            u .= ground_state(u, Ω, gtol);
            Ju = dot(u, J(u)) |> real
            ru = abs(find_vortex(u))
            if Ju < 0.1	# outside
                Ωs[2,j] = Ω
                rs[2,j] = r₁
                break
            elseif Ju > 0.9Jmax		# inside
                Ωs[1,j] = Ω
                rs[1,j] = r₁
                break
            elseif abs(ru-r₀) ≥ abs(ru-rv)
                side = ru < r₀ ? 1 : 2
                Ωs[side,j] = Ω
                rs[side,j] = r₁
                break
            end
            rv = ru
        end
    
        0.1 < real(dot(u, J(u))) < 0.9Jmax  &&  break
    end
    
    mean(Ωs), u
end
