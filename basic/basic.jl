
using LinearAlgebra, BandedMatrices, Optim, DifferentialEquations, JLD2
using Plots, ComplexPhasePortrait

function crds(u)
    N = size(u,1)
    h = 0.6/√N
    h/2*(1-N:2:N-1)
end
    
zplot(ψ) = plot(crds(ψ), crds(ψ), portrait(reverse(ψ,dims=1)).*abs2.(ψ)/maximum(abs2.(ψ)), aspect_ratio=1)
zplot(ψ::Matrix{<:Real}) = zplot(Complex.(ψ))
argplot(ψ) = plot(crds(ψ), crds(ψ), portrait(reverse(ψ,dims=1)), aspect_ratio=1)
argplot(ψ::Matrix{<:Real}) = argplot(Complex.(ψ))

# @load "basic.jld2" Es φas S0 S1 S3 S10

sqnc(j) = [S0[j], S1[j], S3[j], S10[j]]
er(j) = S[j] - dot(S[1], S[j])*S[1]
sqplot(j) = plot(zplot.(sqnc(j))..., argplot.(sqnc(j))...,
    zplot.(er(u, j) for u in sqnc(j))...,
    layout=@layout [a b c d; e f g h; i j k l])

ers(j) = norm.(er(u, j) for u in sqnc(j))

# ssh -l rpolking ozstar.swin.edu.au 'cd /fred/oz127/rpolking/BAB; sbatch runjob'