# Fermion sound wave modes for a Thomas-Fermi cloud

using LinearAlgebra, BandedMatrices, JLD2

zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)

g = 10;  E_f = 20;  Ω = 2*0.2
# h = 0.35;  N = 26
h = 0.5;  N = 10
a = 1.4;  b = 0.2		# SOR and thermal polations

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)

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
∇² = kron(eye, ∂²) + kron(∂², eye)
H = -∇²/4/E_f + diagm(0=>V[:]) - E_f*Matrix(I,N^2,N^2)
J = 1im*(repeat(y,1,N)[:].*kron(∂,eye)-repeat(x,N,1)[:].*kron(eye,∂))

# iterate to self-consistent thermal cloud

Δ = zero(ψ)
gap = []

for _ = 1:2
    global Δ, ev, ew
    
    # sound wave spectrum
    
    Q = diagm(0=>Δ[:])
    s = eigen([
        H+R-Ω*J    Q;
        conj.(Q)    -H-R-Ω*J
    ])
    ev = s.vectors ./ h
    ew = s.values
    
    f = sign.(ew-E_f)
#    Δ += b*(g*reshape(sum(abs2.(ev[N^2+1:end,2:end]), dims=2), size(ψ)) - Δ)
#    push!(gap, Δ)
end

L = (ψ[:]'*J*ψ[:])/norm(ψ)^2

Jev = collect(real(ev[:,i]'*[J zero(J);  zero(J) J]*ev[:,i]-L/h^2)/norm(ev[:,i])^2 for i = 1:length(ew))

@save "basout.jld2" y ev ew oprm gap rsdls

if false

Q = diagm(0=>ψ[:]);  R = 2C*(abs2.(Q)+diagm(0=>nnc[:]))
B = [
        H+R-Ω*J    C*Q.^2;
        -C*conj.(Q).^2    -H-R-Ω*J
]
prob = ODEProblem(s->B*s, ev, (0.0,1.0))
sol = solve(prob)

end