# BdG mode dynamics for an offset vortex

using LinearAlgebra, BandedMatrices, DifferentialEquations
using Plots, ComplexPhasePortrait, Printf

C = 10;  μ = 15;  Ω = 2*0
h = 0.2;  N = 50
# h = 0.2;  N = 30
a = 1.4;  b = 0.2		# SOR and thermal polations

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)
k₀ = Tuple(keys(r²)[argmin(abs.(z.-0.5))])
ψ = Complex.(exp.(-r²/2)/√π);  ψ = conj.(z).*ψ./sqrt(1 .+ r²)
# jitter to include L ≠ 0 component
ψ += (0.1*randn(N,N) + 0.1im*randn(N,N)).*abs.(ψ)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
∂² = (1/h^2).*op([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])

# find ground state solve by SOR: 
# -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ

function impose_vortex!(ψ)
    # boundary condition ψ(z[k₀]) = 0, ψ is analytic at z[k₀]
    udlr = [(1,0),(0,1),(-1,0),(0,-1)]
    ψ[CartesianIndex(k₀)] = 0
    α = sum(ψ[CartesianIndex(k₀.+l)]/(l⋅(1,1im)) for l = udlr) / 4
    for l = udlr
        ψ[CartesianIndex(k₀.+l)] = α*(l⋅(1,1im))
    end
end

# components of BdG matrix

eye = Matrix(I,N,N)
H = -kron(eye, ∂²) - kron(∂², eye) + diagm(0=>V[:]) - μ*Matrix(I,N^2,N^2)
J = 1im*(repeat(y,1,N)[:].*kron(∂,eye)-repeat(x,N,1)[:].*kron(eye,∂))

    # solve by SOR: 
    # -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ

Umd(i) = reshape(ev[1:N^2,i], N, N)
Vmd(i) = reshape(ev[N^2+1:end,i], N, N)

ψ₀ = similar(ψ)
nnc = zero(ψ)
    potential = []
    residual = []
    for _ = 1:Int(1000)
        ψ₀ .= ψ
        for k = keys(ψ)
            i,j = Tuple(k)
            ψ[k] = 0
            T = ∂²[i:i,:]*ψ[:,j:j]+ψ[i:i,:]*∂²[:,j:j]
            L = y[i]*(ψ[i:i,:]*∂'[:,j:j])-x[j]*(∂[i:i,:]*ψ[:,j:j])
            ψk = (μ*ψ₀[k]+T[]+1im*Ω*L[]) /
                (-2*∂²[1,1]+V[k]+C*(abs2.(ψ₀[k])+2nnc[k]))
            ψ[k] = ψ₀[k] + a*(ψk-ψ₀[k])
            impose_vortex!(ψ)
         end
         Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*(abs2.(ψ₀)+2nnc).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
         # take residual of unconstrained components
         impose_vortex!(Lψ)
         E = sum(conj.(ψ).*Lψ)/norm(ψ)^2 |> real
         push!(potential, E)
         push!(residual, norm(Lψ-E*ψ)/norm(ψ))
    end

# propagate

P = ODEProblem((ψ,p,t)->-1im*(-∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))), ψ, (0.0,1.0))
S = solve(P)
ψ = S.u[end]

    # sound wave spectrum
    
Umd(i) = reshape(ev[1:N^2,i], N, N)
Vmd(i) = reshape(ev[N^2+1:end,i], N, N)

    Q = diagm(0=>ψ[:]);  R = 2C*(abs2.(Q)+diagm(0=>nnc[:]))
    BdGmat = [
        H+R-Ω*J    C*Q.^2;
        -C*conj.(Q).^2    -H-R-Ω*J
    ]
    s = eigen(BdGmat)
    ew = s.values
    ixs = eachindex(ew)[real.(ew) .> 1.0]
    sort!(ixs, by=i->real(ew[i]))
    ixs = ixs[1:20]
    ew = ew[ixs];  ev = s.vectors[:,ixs]
    ωs = real.(ew)
Umd(i) = reshape(ev[1:N^2,i], N, N)
Vmd(i) = reshape(ev[N^2+1:end,i], N, N)
    nsq = [h^2*sum(abs2.(Umd(i))-abs2.(Vmd(i))) for i = eachindex(ωs)]
    ev = ev./sqrt.(abs.(nsq))'
    zeromode = [ψ[:]; ψ[:]]
    ev = [zeromode ev]

# thermal cloud

nnc = 0.2*reshape(sum(abs2.(ev[N^2+1:end,2:end]), dims=2), size(ψ))

# mode dynamics

R = 2C*(abs2.(Q)+diagm(0=>nnc[:]))
    BdGmat = [
        H+R-Ω*J    C*Q.^2;
        -C*conj.(Q).^2    -H-R-Ω*J
    ]
P2 = ODEProblem((ev,p,t)->BdGmat*ev, ev, (0.0,0.1))
S2 = solve(P2)
ev2 = S2.u[end]

L = (ψ[:]'*J*ψ[:])/norm(ψ)^2

Jev = collect(real(ev[:,i]'*[J zero(J);  zero(J) J]*ev[:,i]-L/h^2)/norm(ev[:,i])^2 for i = 1:length(ωs))

Umd(i) = reshape(ev[1:N^2,i], N, N)
Vmd(i) = reshape(ev[N^2+1:end,i], N, N)
U2md(i) = reshape(ev2[1:N^2,i], N, N)
V2md(i) = reshape(ev2[N^2+1:end,i], N, N)

nsq = [h^2*sum(abs2.(Umd(i))-abs2.(Vmd(i))) for i = eachindex(ωs)]

zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))

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

function cmpmode(i)
	U = zplot(Umd(i))
	title!(U, "u0")
	V = zplot(Vmd(i))
	title!(V, @sprintf("%.1e * v0*", norm(Umd(i))/norm(Vmd(i))))
	U2 = zplot(U2md(i))
	V2 = zplot(V2md(i))
	plot(U, V, U2, V2, layout=@layout [a b; c d])
end
