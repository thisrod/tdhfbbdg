# Compute Berry phases and curvatures for rotated BdG modes

using LinearAlgebra, JLD2
using Superfluids
using Superfluids: bdg_output

include("berry_utils.jl")

R = 10
d = FourierDiscretisation{2}(100, 2R/99) |> Superfluids.default!

@load "modes.jld2"

# Interpolate only inner circle
z = argand(d)
z_in = z[@. abs(z) < R]

function rotated(q, θ)
    u = zero(z)
    u[@. abs(z) < R] = [interpolate(d, q, w*exp(-1im*θ)) for w in z_in]
    u
end

hh = (0:40)*2π/40
rot(u::Array) = [rotated(normalize(u),h) for h in hh]

function select(n)
    global Ω, q, rv, ws, us, vs, ja, jb, jc
    Ω = Ωs[n]
    q = qs[n]
    rv = rr[n]
    ew = ews[n]
    ev = evs[n]
    
    try
        ws, us, vs = bdg_output(d, ew, ev)
    catch e
        ws, us, vs = bdg_output(d, ew, ev, safe=false)
        ix = @. norm(us) > norm(vs)
        ix[1] = !ix[1]
        ws = ws[ix]
        us = us[ix]
        vs = vs[ix]
    end
    ja = argmin(@. abs(ws-1+Ω))
    jb = argmin(@. abs(ws-2))
    jc = argmin(@. abs(ws-1-Ω))
end

bpus = fill(NaN, length(rr))
bpvs = fill(NaN, length(rr))
pcius = Array{Any}(undef,length(rr), 4)
pcivs = Array{Any}(undef,length(rr), 4)
for n = eachindex(rr)
    select(n)
    for j in [1, ja, jb, jc]
        ur = rot(us[j])
        pcius[n, j] = pci(ur)
        bpus[n] = bphase(ur)[end]
        vr = rot(vs[j])
        pcivs[n, j] = pci(vr)
        bpvs[n] = bphase(vr)[end]
    end
end

@save "/fred/oz127/rpolking/rpcis.jld2" rr bpus bpvs pcius pcivs
