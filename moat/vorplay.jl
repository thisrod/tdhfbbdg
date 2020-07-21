# Playing with Wannier derivatives and vortex detection

using JLD2
@load "mob.jld2"
include("../system.jl")
include("../figs.jl")

# fudge dynamic phase
ffs = [mean(@. Su[j]*(r>R+w)) for j = eachindex(Su)]
@. Su /= (ffs/abs(ffs))