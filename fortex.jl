# Order parameter for an offset Bose vortex, by forcing boundary conditions

using LinearAlgebra, BandedMatrices
using Plots, ComplexPhasePortrait, Printf

zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)

C = 10;  μ = 10;  Ω = 2*0
h = 0.5;  N = 11
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

# solve by SOR: 
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

# iterate to self-consistent thermal cloud

nnc = zero(ψ)
ψ₀ = similar(ψ)
therm = []
oprm = []
rsdls = []

for _ = 1:1
    global nnc, ev, ew

    # solve by SOR: 
    # -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ
    
    potential = []
    residual = []
    for _ = 1:Int(1e3)
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
    
    # sound wave spectrum
    
    Q = diagm(0=>ψ[:]);  R = 2C*(abs2.(Q)+diagm(0=>nnc[:]))
    s = eigen([
        H+R-Ω*J    C*Q.^2;
        -C*conj.(Q).^2    -H-R-Ω*J
    ])
    ev = s.vectors
    nsq = h^2*sum(abs2.(ev[1:N^2,:])-abs2.(ev[N^2+1:end,:]), dims=1)
    ixs = nsq[:].>0
    ew = real.(s.values)[ixs];  ev = ev[:,ixs];  nsq = nsq[:,ixs]
    σ = sortperm(ew)
    ew = ew[σ];  ev = ev[:,σ];  nsq = nsq[:,σ]
    ev[:,2:end] = ev[:,2:end]./sqrt.(nsq[:,2:end])
    ev[:,1] *= norm(ψ)

    nnc += b*(reshape(sum(abs2.(ev[N^2+1:end,2:end]), dims=2), size(ψ)) - nnc)
    push!(oprm, copy(ψ₀))
    push!(therm, nnc)
    push!(rsdls, residual)
end

L = (ψ[:]'*J*ψ[:])/norm(ψ)^2

ωs = ew[1:100]
ev = ev[:,1:100]

Jev = collect(real(ev[:,i]'*[J zero(J);  zero(J) J]*ev[:,i]-L/h^2)/norm(ev[:,i])^2 for i = 1:length(ωs))

# @save "basout.jld2" y ev ew oprm therm rsdls

Umd(i) = reshape(ev[1:N^2,i], N, N)
Vmd(i) = reshape(ev[N^2+1:end,i], N, N)

zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)

function showmode(i)
	M = scatter(Jev, ωs ./ 2, mc=:black, ms=2, msw=0, leg=:none)
	scatter!(M, Jev[i:i], ωs[i:i] / 2, mc=:red, ms=4, msw=0, leg=:none)
	title!(M, @sprintf("w = %.4f, J = %.3f", ωs[i]/2, Jev[i]))
	U = zplot(Umd(i))
	title!(U, "u")
	V = zplot(Vmd(i))
	title!(V, @sprintf("%.1e * v*", norm(Umd(i))/norm(Vmd(i))))
	plot(M, U, V, layout=@layout [a; b c])
end

if false

Q = diagm(0=>ψ[:]);  R = 2C*(abs2.(Q)+diagm(0=>nnc[:]))
B = [
        H+R-Ω*J    C*Q.^2;
        -C*conj.(Q).^2    -H-R-Ω*J
]
prob = ODEProblem(s->B*s, ev, (0.0,1.0))
sol = solve(prob)

end
