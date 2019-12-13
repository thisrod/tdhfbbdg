# Single vortex diagonalisation benchmark

using LinearAlgebra, BandedMatrices, SparseArrays, JLD2, Arpack
using Plots, ComplexPhasePortrait, Printf

C = 10;  μ = 10;  Ω = 2*0.575
h = 0.2;  N = 40
rdtol = 1e-5	# residual tolerance
a = 1.4		# SOR polation

y = h/2*(1-N:2:N-1);  x = y';  z = Complex.(x,y)
V = r² = abs2.(z)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

memoized = false
memfile = "bnop.jld2"
if isfile(memfile)
    @load memfile ψ prms residual
    memoized =
        size(ψ) == (N,N) &&
        residual < rdtol &&
        prms == Dict(:C=>C, :μ => μ, :Ω=>Ω, :h=>h)
end

if memoized
    println("Order parameter: here's one I prepared earlier")
else

    # Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x
    
    function op(stencil)
        mid = (length(stencil)+1)÷2
        diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
        BandedMatrix(Tuple(diags), (N,N))
    end
    ∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
    ∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
    
    # solve by SOR: 
    # -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ
    
    ψ = Complex.(exp.(-r²/2)/√π);  ψ = conj.(z).*ψ./sqrt(1 .+ r²)
    ψ₀ = similar(ψ)
    println("Order parameter by SOR")
    n = 0;  residual = Inf
    @time while n < 5000 && residual > rdtol
        global residual, n
        ψ₀ .= ψ
        for k = keys(ψ)
            i,j = Tuple(k)
            ψ[k] = 0
            T = ∂²[i:i,:]*ψ[:,j:j]+ψ[i:i,:]*∂²[:,j:j]
            L = y[i]*(ψ[i:i,:]*∂'[:,j:j])-x[j]*(∂[i:i,:]*ψ[:,j:j])
            ψk = (μ*ψ₀[k]+T[]+1im*Ω*L[]) /
                (-2*∂²[1,1]+V[k]+C*abs2.(ψ₀[k]))
            ψ[k] = ψ₀[k] + a*(ψk-ψ₀[k])
         end
         n += 1
         if n % 100 == 0
             Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ₀).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
             E = sum(conj.(ψ).*Lψ)/norm(ψ)^2 |> real
             residual = norm(Lψ-E*ψ)/norm(ψ)
         end
    end
    
   prms = Dict(:C=>C, :μ => μ, :Ω=>Ω, :h=>h)
    @save memfile ψ prms residual

end

# Dense BdG matrix

eye = Matrix(I,N,N)
J = 1im*(repeat(y,1,N)[:].*kron(∂,eye)-repeat(x,N,1)[:].*kron(eye,∂))
H = -kron(eye, ∂²) - kron(∂², eye) + diagm(0=>V[:]) - μ*Matrix(I,N^2,N^2)
Q = diagm(0=>ψ[:]);  R = 2C*abs2.(Q)
const BdGmat = [
    H+R-Ω*J    C*Q.^2;
    -C*conj.(Q).^2    -H-R-Ω*J
]
const trial_uv = [ψ[:]; ψ[:]]

# Julia sparse matrices

const BdGsparse = sparse(BdGmat)

# matmul benchmarks

function apply_op(M,n)
    for _ = 1:n
        M*trial_uv
    end
end

function apply_op!(M,n)
    out = similar(trial_uv)
    for _ = 1:n
        mul!(out, M, trial_uv)
    end
end

# Use eigen(M) for brute force spectrum

# Iterative spectrum for n eigenvalues

function spectrum(M,n)
    eigs(M; nev=n, which=:SM) 
end

# # spectrum by brute force
#     
# println("Dense matrix and brute force")
# @time s = eigen(BdGmat)
# ev = s.vectors
# # This way seems to beat the rounding
# unms = h^2*sum(abs2.(ev[1:N^2,:]), dims=1)
# vnms = h^2*sum(abs2.(ev[N^2+1:end,:]), dims=1)
# nsq = unms .- vnms
# ixs = sortperm(nsq[:], by=abs)
# ixs = ixs[1:10]
# ωs = s.values[ixs]
# 
# # L = (ψ[:]'*J*ψ[:])/norm(ψ)^2
# 
# # Jev = collect(real(ev[:,i]'*[J zero(J);  zero(J) J]*ev[:,i]-L/h^2)/norm(ev[:,i])^2 for i = 1:length(ωs))
# 
# Umd(i) = reshape(ev[1:N^2,i], N, N)
# Vmd(i) = reshape(ev[N^2+1:end,i], N, N)
# 
# # nsq = [h^2*sum(abs2.(Umd(i))-abs2.(Vmd(i))) for i = eachindex(ωs)]
# 
# zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
# zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
# 
# function showmode(i)
# 	M = scatter(Jev[nsq[:].≥0], real.(ωs[nsq[:].≥0]) ./ 2, mc=:black, ms=3, msw=0, leg=:none)
# 	scatter!(M, Jev[nsq[:].<0], real.(ωs[nsq[:].<0]) ./ 2, mc=:green, ms=3, msw=0, leg=:none)
# 	scatter!(M, Jev[i:i], real.(ωs[i:i]) / 2, mc=:red, ms=4, msw=0, leg=:none)
# 	title!(M, @sprintf("%.0f, w = %.4f, J = %.3f", nsq[i], real(ωs[i])/2, Jev[i]))
# 	U = zplot(Umd(i))
# 	title!(U, "u")
# 	V = zplot(Vmd(i))
# 	title!(V, @sprintf("%.1e * v*", norm(Umd(i))/norm(Vmd(i))))
# 	plot(M, U, V, layout=@layout [a; b c])
# end
