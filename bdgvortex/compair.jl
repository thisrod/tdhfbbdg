# Compare Berry phases for offset vortex and pair

using LinearAlgebra, JLD2, Printf, Plots
using Superfluids
using Superfluids: bdg_output

R = 10
s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2) |> Superfluids.default!
d = FourierDiscretisation{2}(100, 2R/99) |> Superfluids.default!

Superfluids.default!(:xlims, (-6,6))
Superfluids.default!(:ylims, (-6,6))

J = Superfluids.operators(:J) |> only
Jev(u) = dot(u, J(u))/sum(abs2,u) |> real
Jev(u,v) =
    (norm2(u)*(Jev(u) - Jev(q)) - norm2(v)*(Jev(v) - Jev(q)))/(norm2(u)+norm2(v))

norm2(u) = sum(abs2, u)

function safe_output(q, ew, ev)
    local ws, us, vs
    try
        ws, us, vs = bdg_output(d, ew, ev)
    catch e
        ws, us, vs = bdg_output(d, ew, ev, safe=false)
        ix = @. norm(us) > norm(vs)
        for j = 1:2:length(ix)
            ix[j] ⊻ ix[j+1] && continue
#            @assert abs(ws[j]) < 1e-7 && abs(ws[j+1]) < 1e-7
            ix[j] = !ix[j]
        end
        ws = ws[ix]
        us = us[ix]
        vs = vs[ix]
    end
    pushfirst!(ws,0)
    pushfirst!(us, q)
    pushfirst!(vs, q)
    ws, us, vs
end


myload(s) = jldopen(s) do f
    rs = f["rr"]
    Ωs = f["Ωs"]
    wss = []
    uss = []
    vss = []
    for (q, ew, ev) in zip(f["qs"], f["ews"], f["evs"])
        ws, us, vs = safe_output(q, ew, ev)
        push!(wss, ws)
        push!(uss, us)
        push!(vss, vs)
    end
    Ωs, wss, uss, vss, rs
end

Ωos, wos, uos, vos, ros = myload("modes.jld2")
Ωps, wps, ups, vps, rps = myload("pairmodes.jld2")
R_TF = ros[end]

function plot_precession()
    plot(ros/R_TF, Ωos, label="orbit", legend=:bottomright)
    plot!(rps/R_TF, Ωps, label="pair")
    xlabel!("rv/R_TF")
    ylabel!("W")
    title!("Precession frequency convergence at large radius")
end

function plot_kelvon()
    P = plot(legend=:bottomright)
    S(rss, uss, j, l) = scatter!(rss/R_TF, [Jev(us[j]) for us in uss], label=l)
    S(ros, vos, 2, "orbit")
    S(rps, vps, 2, "pair -")
    S(rps, vps, 3, "pair +")
    xlabel!("rv/R_TF")
    ylabel!("<J>")
    title!("Kelvon v modes")
end