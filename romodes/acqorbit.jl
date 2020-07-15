# Acquire a rotating vortex orbit by bisection search

source = open(@__FILE__) do f
    read(f, String)
end

using LinearAlgebra, BandedMatrices, Optim
using Statistics: mean

C = 3000
N = 100
l = 20.0	# maximum domain size
# dts = 10 .^ (-5:-0.5:-9.0)	# residual

include("../system.jl")

# r₀ = 1.7
# r₀ = 2.5		# vortex orbit radius
r₀ = 3.2

# r₀, Ω, φ = acquire_orbit(r₀, 1e-6)
# ψ = ground_state(φ, Ω, 1e-6)
