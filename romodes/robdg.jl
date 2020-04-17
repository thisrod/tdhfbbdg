# Trial of BdGMatrices

using BdGMatrices
using LinearAlgebra, BandedMatrices, Arpack, Optim, DifferentialEquations, JLD2
using Plots, ComplexPhasePortrait, Printf

h = 0.119;  N = 108
C = 2748.85

@load "romodes.jld2" Ω psi

A = BdGMatrix(C, Ω, N, h)

# Manually add old ground-state solution

A.ψ .= psi
let L = BdGMatrices.Lop(A)
    A.V .-= sum(conj.(A.ψ).*L(A.ψ)) |> real
end
   
# Find Kelvin mode

# ωs,uvs,nconv,niter,nmult,resid = eigs(BdGmat; nev=16, which=:SM) 

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
