# Vortex dynamics without abstraction, just vectors and matrices.

using LinearAlgebra, BandedMatrices, JLD2

zplot(ψ) = plot(y, y, portrait(ψ).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)

C = 10;  μ=20;  Ω=2*0.2
h = 0.35;  N = 26
a = 1.4;  b = 0.2		# SOR and thermal polations

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)
ψ = Complex.(exp.(-r²/2)/√π);  ψ = z.*ψ./sqrt(1 .+ r²)

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

for _ = 1:2
    global nnc, ev, ew

    # solve by SOR: 
    # -∂²*ψ-ψ*∂²+V.*ψ+C*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ)) = μ*ψ
    
    rsdl = []
    for _ = 1:200
        copy!(ψ₀, ψ)
        for k = keys(ψ)
            i,j = Tuple(k)
            ψ[k] = 0
            T = ∂²[i:i,:]*ψ[:,j:j]+ψ[i:i,:]*∂²[:,j:j]
            L = y[i]*(ψ[i:i,:]*∂'[:,j:j])-x[j]*(∂[i:i,:]*ψ[:,j:j])
            ψk = (μ*ψ₀[k]+T[]+1im*Ω*L[]) /
                (-2*∂²[1,1]+V[k]+C*(abs2.(ψ₀[k])+2nnc[k]))
            ψ[k] = ψ₀[k] + a*(ψk-ψ₀[k])
         end
         Lψ = -∂²*ψ-ψ*∂²+V.*ψ+C*(abs2.(ψ₀)+2nnc).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
         m = sum(conj.(ψ).*Lψ)/norm(ψ)^2
         push!(rsdl, norm(Lψ-m*ψ)/norm(ψ))
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
    push!(rsdls, rsdl)
end

L = (ψ[:]'*J*ψ[:])/norm(ψ)^2

Jev = collect(real(ev[:,i]'*[J zero(J);  zero(J) J]*ev[:,i]-L/h^2)/norm(ev[:,i])^2 for i = 1:length(ew))

@save "basout.jld2" y ev ew oprm therm rsdls

if false

Q = diagm(0=>ψ[:]);  R = 2C*(abs2.(Q)+diagm(0=>nnc[:]))
B = [
        H+R-Ω*J    C*Q.^2;
        -C*conj.(Q).^2    -H-R-Ω*J
]
prob = ODEProblem(s->B*s, ev, (0.0,1.0))
sol = solve(prob)

end