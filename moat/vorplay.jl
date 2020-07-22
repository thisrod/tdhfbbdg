# Moat vortex detection and Wu phases

using JLD2
@load "mob.jld2"
include("../system.jl")
include("../figs.jl")

# fudge dynamic phase
ffs = [mean(@. Su[j]*(r>R+w)) for j = eachindex(Su)]
@. Su /= (ffs/abs(ffs))

ins = [find_vortex(q) for q in Su]
outs = [find_moat(q) for q in Su]

tix(t) = argmin(@. abs(St-t))