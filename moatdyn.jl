# BdG mode dynamics for an offset vortex with moat

using LinearAlgebra, BandedMatrices, DifferentialEquations, Optim, Arpack

C = 250.0
Ω = 0.0
R = 1.3
w = 0.15	# moat width
# ω = -2.86	# potential offset outside moat for lock step
# ω = -10.0	# potential offset outside moat for fast vortex
ω = 0.0

h = 0.1
N = 60
# h = 0.3
# N = 20


y = h/2*(1-N:2:N-1);  x = y';  z = x .+ 1im*y
r = abs.(z)
r² = abs2.(z)

V = r²  + 20*R^2*exp.(-(r.-R).^2/2/w^2)
W = V .+ ω*(r.>R)

# Finite difference matrices.  ∂ on left is ∂y, ∂' on right is ∂x

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end

# Hard zero boundary conditions.
∂ = (1/h).*op([0, -1/2, 0, 1/2, 0])
∂² = (1/h^2).*op(Float64[0, 0, 1, -2, 1, 0, 0])

# Minimise the energy 
#
# E(ψ) = -∫ψ*∇²ψ/2 + V|ψ|²+g/2·|ψ|⁴
#
# The GPE functional L(ψ) is the gradient required by Optim.

L(ψ) = -(∂²*ψ+ψ*∂²')/2+V.*ψ+C/h*abs2.(ψ).*ψ
Ham(ψ) = -(∂²*ψ+ψ*∂²')/2+V.*ψ+C/2h*abs2.(ψ).*ψ
E(xy) = sum(conj.(togrid(xy)).*Ham(togrid(xy))) |> real
grdt!(buf,xy) = copyto!(buf, 2*L(togrid(xy))[:])
togrid(xy) = reshape(xy, size(z))

rc = 2	# condensate radius, 
P = Diagonal(sqrt.(rc^4 .+ V[:].^2))

# TODO relax residual tolerance
init = z.*(r .< (R-w))
result = optimize(E, grdt!, z[:],
    GradientDescent(manifold=Sphere()),
    Optim.Options(iterations = 10000, allow_f_increases=true)
)
ψ = togrid(result.minimizer)
φ = copy(ψ)
ψ[abs.(z) .> R] .= 0

# components of BdG matrix

μ = sum(conj.(ψ).*L(ψ)) |> real
eye = Matrix(I,N,N)
H = -kron(eye, ∂²)/2 - kron(∂², eye)/2 + diagm(0=>V[:]) - μ*Matrix(I,N^2,N^2)
Q = diagm(0=>ψ[:])
    BdGmat = [
        H+2C/h*abs2.(Q)    C/h*Q.^2;
        -C/h*conj.(Q).^2    -H-2C/h*abs2.(Q)
    ]

# Find Kelvin mode

ωs,uvs,nconv,niter,nmult,resid = eigs(BdGmat; nev=16, which=:SM) 

umode(j) = reshape(uvs[1:N^2, j], N, N)
vmode(j) = reshape(uvs[N^2+1:end, j], N, N)

k = 2		# known Kelvin mode
bnum = N*sum(abs2.(umode(k)) - abs2.(vmode(k)))
uk = umode(k) ./ √bnum
vk = vmode(k) ./ √bnum

φ += uk + conj.(vk)
uv = [φ[:] uk[:]; φ[:] vk[:]]

# kludge BC
∂[1,:] .= ∂[2,:]
∂[end,:] .= ∂[end-1,:]
∂²[1,:] .= ∂²[2,:]
∂²[end,:] .= ∂²[end-1,:]

# mode dynamics

μ = sum(conj.(φ).*L(φ)) |> real
H = -kron(eye, ∂²)/2 - kron(∂², eye)/2 + diagm(0=>W[:]) - μ*Matrix(I,N^2,N^2)
function BGM(uv) 
    φ = reshape(uv[1:N^2,1], N, N)
    Q = diagm(0=>φ[:])
    [
        H+2C/h*abs2.(Q)    C/h*Q.^2;
        -C/h*conj.(Q).^2    -H-2C/h*abs2.(Q)
    ]
end    
    
P = ODEProblem((ev,p,t)->BGM(uv)*uv, uv, (0.0,0.2), saveat=0.05)
S = solve(P)

# L = (ψ[:]'*J*ψ[:])/norm(ψ)^2

# Jev = collect(real(ev[:,i]'*[J zero(J);  zero(J) J]*ev[:,i]-L/h^2)/norm(ev[:,i])^2 for i = 1:length(ωs))
