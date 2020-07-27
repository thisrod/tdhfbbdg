# Acquire a rotating vortex orbit by bisection search

source = open(@__FILE__) do f
    read(f, String)
end

using LinearAlgebra, BandedMatrices, Optim
using Statistics: mean

C = 3000
N = 400
l = 20.0	# maximum domain size

include("../system.jl")

r₀ = 2.0

u = φ
r₀, Ω, φ = acquire_orbit(r₀, 1e-6)
ψ = ground_state(u, Ω, 1e-6)
