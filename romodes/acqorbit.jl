# Acquire a rotating vortex orbit by bisection search

source = open(@__FILE__) do f
    read(f, String)
end

using LinearAlgebra, BandedMatrices, Optim, JLD2
using Statistics: mean

C = 3000
N = 100
l = 20.0	# maximum domain size
# dts = 10 .^ (-5:-0.5:-9.0)	# residual

include("../system.jl")

# r₀ = 1.7
# r₀ = 2.5		# offset of imprinted vortex
r₀ = 3.9

# Ω, φ = acquire_orbit(r₀, 1e-6)
# ψ = ground_state(φ, Ω, 1e-6)

# @save "acqorbit2.jld2" source Ω φ ψ
