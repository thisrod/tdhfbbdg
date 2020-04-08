# Dynamical Kelvin mode for an offset vortex

using LinearAlgebra, BandedMatrices, Arpack, Optim, DifferentialEquations
using Plots, ComplexPhasePortrait, Printf

Nc = 5000
# h = 0.2523;  N = 36
h = 0.178;  N = 72
C = 2748.85
# Ω= 0.21213
Ω= (0.2875+0.29375)/2

r₀ = 1.9		# offset of imprinted phase

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

# Minimise the energy 
#
# E(ψ) = -∫ψ*∇²ψ/2 + V|ψ|²+g/2·|ψ|⁴
#
# The GPE functional L(ψ) is the gradient required by Optim.

L(ψ) = -(∂²*ψ+ψ*∂²')/2+V.*ψ+C/h*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
Ham(ψ) = -(∂²*ψ+ψ*∂²')/2+V.*ψ+C/2h*abs2.(ψ).*ψ-1im*Ω*(y.*(ψ*∂')-x.*(∂*ψ))
E(xy) = sum(conj.(togrid(xy)).*Ham(togrid(xy))) |> real
grdt!(buf,xy) = copyto!(buf, 2*L(togrid(xy))[:])
togrid(xy) = reshape(xy, size(z))

rc = 2	# condensate radius, 
P = Diagonal(sqrt.(rc^4 .+ V[:].^2))

result = optimize(E, grdt!, ψ[:],
    GradientDescent(manifold=Sphere()),
    Optim.Options(iterations = 200_000, allow_f_increases=true)
)
ψ = togrid(result.minimizer)
   
# sound wave spectrum

μ = sum(conj.(ψ).*L(ψ)) |> real
eye = Matrix(I,N,N)
H = -kron(eye, ∂²)/2 - kron(∂², eye)/2 + diagm(0=>V[:]) - μ*Matrix(I,N^2,N^2)
J = 1im*(repeat(y,1,N)[:].*kron(∂,eye)-repeat(x,N,1)[:].*kron(eye,∂))
Q = diagm(0=>ψ[:])
    BdGmat = [
        H+2C/h*abs2.(Q)-Ω*J    -C/h*Q.^2;
        C/h*conj.(Q).^2    -H-2C/h*abs2.(Q)-Ω*J
    ]

#    nnc += b*(reshape(sum(abs2.(ev[N^2+1:end,2:end]), dims=2), size(ψ)) - nnc)
#    push!(oprm, copy(ψ₀))
#    push!(therm, nnc)

# Find Kelvin mode

ωs,uvs,nconv,niter,nmult,resid = eigs(BdGmat; nev=16, which=:SM) 

umode(j) = reshape(uvs[1:N^2, j], N, N)
vmode(j) = reshape(uvs[N^2+1:end, j], N, N)
umode(j,n) = reshape(S[n][1:N^2, j], N, N)
vmode(j,n) = reshape(S[n][N^2+1:end, j], N, N)

# kk = 3		# Kelvin mode found by eyeball
# bnum = Nc*sum(abs2.(umode(kk)) - abs2.(vmode(kk)))
# 
# init = [[ψ[:]; conj.(ψ[:])] uvs[:,kk]/√bnum]
# 
# # TODO kludge BC
# 
# # mode dynamics in lab frame
# 
# function BGM(ψ)
#     @assert size(ψ) == (N,N)
#     Q = diagm(0=>ψ[:])
#     [
#         H+2C/h*abs2.(Q)    -C/h*Q.^2;
#         C/h*conj.(Q).^2    -H-2C/h*abs2.(Q)
#     ]
# end
# 
# function deriv(A,_,_)
#     # first column is order parameter, following GPE
#     ψ = reshape(A[1:N^2, 1], N, N)
#     uv = A[:,2:end]
#     dψ = -1im*(-(∂²*ψ+ψ*∂²)/2+(V.-μ).*ψ+C/h*abs2.(ψ).*ψ)
#     duv = BGM(ψ)*uv
#     [[dψ[:]; conj.(dψ[:])] duv]
# end
# 
# P = ODEProblem(deriv, init, (0.0,0.1), saveat=0.01)
# S = solve(P)
# # 
# # function showmode(i)
# # 	M = scatter(Jev[nsq[:].≥0], real.(ωs[nsq[:].≥0]) ./ 2, mc=:black, ms=3, msw=0, leg=:none)
# # 	scatter!(M, Jev[nsq[:].<0], real.(ωs[nsq[:].<0]) ./ 2, mc=:green, ms=3, msw=0, leg=:none)
# # 	scatter!(M, Jev[i:i], real.(ωs[i:i]) / 2, mc=:red, ms=4, msw=0, leg=:none)
# # 	title!(M, @sprintf("%.0f, w = %.4f, J = %.3f", nsq[i], real(ωs[i])/2, Jev[i]))
# # 	U = zplot(Umd(i))
# # 	title!(U, "u")
# # 	V = zplot(Vmd(i))
# # 	title!(V, @sprintf("%.1e * v*", norm(Umd(i))/norm(Vmd(i))))
# # 	plot(M, U, V, layout=@layout [a; b c])
# # end

uvplot(j) = plot(zplot(umode(j)), zplot(vmode(j)), layout=@layout [a b])
uvplot(j,n) = plot(zplot(umode(j,n)), zplot(vmode(j,n)), layout=@layout [a b])
uvplot() = plot(uvplot(1), uvplot(2), uvplot(3), layout=@layout [a;b;c])
splot(n) = plot(uvplot(1,n), uvplot(2,n), layout=@layout [a;b])

zplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
argplot(ψ) = plot(x[:], y, portrait(reverse(ψ,dims=1)), aspect_ratio=1)
argplot(ψ::Matrix{<:Real}) = argplot(Complex.(ψ))
