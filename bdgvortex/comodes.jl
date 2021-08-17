# Combine multiple JLD2 files.  Run this on OzStar

using JLD2

cd("/fred/oz127/rpolking/BVmodes")

vars = [:rr, :ws, :qs, :ews, :evs]
uars = [Symbol(uppercase(string(v))) for v in vars]

for u in uars
    @eval $u = []
end

for file in readdir()
    global rr, ws, qs, ews, evs
    @load file rr ws qs ews evs
    for (v, u) in zip(vars, uars)
        @eval $u = [$u; $v]
    end
end

# unique sort
ix = sortperm(RR)
ix = [j for (i,j) in pairs(ix) if (j == 1 || RR[j] ≉ RR[ix[i-1]])]

for (v, u) in zip(vars, uars)
    @eval $v = $u[ix]
end
Ωs = ws

@save "../modes.jld2" rr Ωs qs ews evs
