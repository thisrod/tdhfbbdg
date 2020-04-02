# Fermion sound wave modes for a Thomas-Fermi cloud

using LinearAlgebra, BandedMatrices, JLD2
using Plots, ComplexPhasePortrait

zplot(ψ) = plot(y, y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)

g = 10;  E_f = 20;  Ω = 2*0.1
# h = 0.35;  N = 26
h = 0.5;  N = 10
b = 0.05		# thermal polation

y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
V = r² = abs2.(z)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x
# TODO develop the missing parts of DiffEqOperators and use that
 
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

Δ = (E_f.>V).*(E_f.-V)^(3/2)/4(π^2)
gap = []
ews = []

for _ = 1:5
    global Δ, ev, ew
    
    # sound wave spectrum
    
    Q = diagm(0=>Δ[:])
    s = eigen([
        H-Ω*J    Q;
        conj.(Q)    -H-Ω*J
    ])
    ev = s.vectors ./ h
    ew = s.values
    
    push!(ews, ew)
    f = ew .< E_f		# zero temperature Fermi-Dirac
    push!(gap, g*reshape(sum(f'.*conj.(ev[1:N^2,:]).*ev[N^2+1:end,:], dims=2), size(V)))
    Δ += b*(gap[end] - Δ)
end

Jev = collect(real(ev[:,i]'*[J zero(J);  zero(J) J]*ev[:,i])/norm(ev[:,i])^2 for i = 1:length(ew));

# scatter(ews[1], mc=:black, ms=2, leg=:none)
# plot!(collect(xlims()), [E_f, E_f], lc=:black, ls=:dashdot)

# scatter(collect(sum(real.(gap[i])) for i=1:5))