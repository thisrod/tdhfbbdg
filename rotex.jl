# BdG modes for an offset vortex in the rotating frame

using LinearAlgebra, BandedMatrices, JLD2
using Plots, ComplexPhasePortrait, Printf

C = 10;  μ=30;  Ω=2*0.15
r₀ = 0.5		# offset of imprinted phase
# C = 10;  μ=30;  Ω=2*0.13
# r₀ = 0.2		# offset of imprinted phase
h = 0.3;  N = 36
a = 1.4		# SOR polation

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)
ψ = Complex.(exp.(-r²/2)/√π);  ψ = (z.-r₀).*ψ

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

# components of BdG matrix

eye = Matrix(I,N,N)
H = -kron(eye, ∂²) - kron(∂², eye) + diagm(0=>V[:]) - μ*Matrix(I,N^2,N^2)
J = 1im*(repeat(y,1,N)[:].*kron(∂,eye)-repeat(x,N,1)[:].*kron(eye,∂))

# iterate to self-consistent thermal cloud

nnc = zero(ψ)
ψ₀ = similar(ψ)
therm = []
oprm = []
rsdls = []
μs = []

memoized = false
memfile = "ofst.jld2"
if isfile(memfile)
    @load memfile ψ prms
    memoized =
        size(ψ) == (N,N) &&
        prms == Dict(:C=>C, :μ => μ, :Ω=>Ω, :h=>h)
end

if memoized
    println("Order parameter: here's one I prepared earlier")
else

    # solve by SOR: 
    # -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ
    
    rsdl = []
    for n = 0:5000
        ψ₀ .= ψ
        for k = keys(ψ)
            i,j = Tuple(k)
            ψ[k] = 0
            T = ∂²[i:i,:]*ψ[:,j:j]+ψ[i:i,:]*∂²[:,j:j]
            L = y[i]*(ψ[i:i,:]*∂'[:,j:j])-x[j]*(∂[i:i,:]*ψ[:,j:j])
            ψk = (μ*ψ₀[k]+T[]+1im*Ω*L[]) /
                (-2*∂²[1,1]+V[k]+C*(abs2.(ψ₀[k])+2nnc[k]))
            ψ[k] = ψ₀[k] + a*(ψk-ψ₀[k])
         end
         Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*(abs2.(ψ)+2nnc).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
         m = sum(conj.(ψ).*Lψ)/norm(ψ)^2
         push!(μs, m)
         push!(rsdl, norm(Lψ-m*ψ)/norm(ψ))
         n % 50 == 0 && push!(oprm, copy(ψ₀))
    end
    
    prms = Dict(:C=>C, :μ => μ, :Ω=>Ω, :h=>h)
    @save memfile ψ prms
end
   
# sound wave spectrum

Q = diagm(0=>ψ[:]);  R = 2C*(abs2.(Q)+diagm(0=>nnc[:]))
s = eigen([
    H+R-Ω*J    C*Q.^2;
    -C*conj.(Q).^2    -H-R-Ω*J
])
ev = s.vectors
nsq = h^2*sum(abs2.(ev[1:N^2,:])-abs2.(ev[N^2+1:end,:]), dims=1)
ixs = nsq[:].>0
@assert maximum(imag.(s.values)) < 1e-5
ωs = real.(s.values)
σ = sortperm(ew[ixs])
ωs = ωs[ixs][σ]
ev = ev[:,ixs][:,σ]
nsq = nsq[ixs][σ]
ev[:,2:end] = ev[:,2:end]./sqrt.(nsq[2:end])'
ev[:,1] *= norm(ψ)

#    nnc += b*(reshape(sum(abs2.(ev[N^2+1:end,2:end]), dims=2), size(ψ)) - nnc)
#    push!(oprm, copy(ψ₀))
#    push!(therm, nnc)

L = (ψ[:]'*J*ψ[:])/norm(ψ)^2

Jev = collect(real(ev[:,i]'*[J zero(J);  zero(J) J]*ev[:,i]-L/h^2)/norm(ev[:,i])^2 for i = 1:length(ew))

Umd(i) = reshape(ev[1:N^2,i], N, N)
Vmd(i) = reshape(ev[N^2+1:end,i], N, N)

zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))

# scatter(rsdls[], ms =2, mc=:black, leg=:none, yscale=:log10)
# Ls = [(ψ[:]'*J*ψ[:])/norm(ψ)^2 |> real for ψ in oprm];

function showmode(i)
	M = scatter(Jev[nsq[:].≥0], real.(ωs[nsq[:].≥0]) ./ 2, mc=:black, ms=3, msw=0, leg=:none)
	scatter!(M, Jev[nsq[:].<0], real.(ωs[nsq[:].<0]) ./ 2, mc=:green, ms=3, msw=0, leg=:none)
	scatter!(M, Jev[i:i], real.(ωs[i:i]) / 2, mc=:red, ms=4, msw=0, leg=:none)
	title!(M, @sprintf("%.0f, w = %.4f, J = %.3f", nsq[i], real(ωs[i])/2, Jev[i]))
	U = zplot(Umd(i))
	title!(U, "u")
	V = zplot(Vmd(i))
	title!(V, @sprintf("%.1e * v*", norm(Umd(i))/norm(Vmd(i))))
	plot(M, U, V, layout=@layout [a; b c])
end
