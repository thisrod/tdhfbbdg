# Energetic stability of vortex rings

# TODO solid where stable, dotted elsewhere

using LinearAlgebra, JLD2, Plots
using Superfluids
using Superfluids: relax, cloud

s = Superfluid{2}(100.0, (x,y)->(x^2+y^2)/2) |> Superfluids.default!
d = FourierDiscretisation{2}(100, 0.121) |> Superfluids.default!

Superfluids.default!(:xlims, (-4,4))
Superfluids.default!(:ylims, (-4,4))
Plots.default(:legend, :none)
Plots.default(:size, (300,200))

J, H = Superfluids.operators(:J, :H)
Jev(u) = dot(u, J(u))/sum(abs2,u) |> real
# Lev calculates 〈H〉, not 〈L〉
Lev(q, Ω) = dot(q, H(q; Ω)) |> real
Lev(q) = Lev(q, 0.0)

S, P, T = distinguishable_colors(3, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

# TODO exclude points with Ω ≈ 0.6
function loadmerge(ss...)
    global rr, Ωs, qs, ews, evs
    RR = [];  WS = [];  QS = []
    for s in ss
        @load s rr Ωs qs
        rr = rr[eachindex(Ωs)]  # jobs that timed out
        RR = [rr; RR]
        ixs = sortperm(RR)
        RR = RR[ixs]
        WS = [Ωs; WS];  WS = WS[ixs]
        QS = [qs; QS];  QS = QS[ixs]
    end
    rr, Ωs, qs = RR, WS, QS
    nothing
end

@load "firmmodes.jld2"
RTF = rr[end]
rs, Ws = rr, Ωs
@assert length(rs) == length(Ws)
Es = [Lev(x...) for x in zip(qs, Ωs)]
Js = Jev.(qs)
Fs = Lev.(qs)

loadmerge("firmpairoprm.jld2", "firmpairouter.jld2")
rp, Wp = rr, Ωs
@assert length(rp) == length(Wp)
Ep = [Lev(x...) for x in zip(qs, Ωs)]
Jp = Jev.(qs)
Fp = Lev.(qs)

loadmerge("firmtripleoprm.jld2", "firmtripleouter.jld2")
rt, Wt = rr, Ωs
@assert length(rt) == length(Wt)
Et = [Lev(x...) for x in zip(qs, Ωs)]
Jt = Jev.(qs)
Ft = Lev.(qs)

Q0 = steady_state()
E0 = Lev(Q0, 0.0)

Wc = [0.7,0.3]

Q1 = steady_state(s, d, rvs=[complex(0.0)], as=[0.0])
E1 = [Lev(Q1, Ω) for Ω in Wc]
F1 = fill(Lev(Q1), 2)

Q2 = steady_state(s, d, initial=cloud(d).*argand(d).^2)
E2 = [Lev(Q2, Ω) for Ω in Wc]
F2 = fill(Lev(Q2), 2)

result = relax(s, d, initial=cloud(d).*argand(d).^3, Ω=0.7, iterations=100)
Q3 = result.minimizer
E3 = [Lev(Q3, Ω) for Ω in Wc]
F3 = fill(Lev(Q2), 2)

leftof(p, q) = maximum(p) < minimum(q)
olap1(p, q) = !(leftof(p, q) || leftof(q, p))
olap2((px, py), (qx, qy)) = (olap1(px, qx) && olap1(py, qy))
cindex(xx) = eachindex(xx)[1:end-1]
couples(xx) = ((xx[j], xx[j+1]) for j = cindex(xx))
bboxes(xx, yy) = zip(cindex(xx), zip(couples.([xx, yy])...))
nears(x1, y1, x2, y2) = [(j, p, k, q) for (j, p) in bboxes(x1, y1) for (k, q) in bboxes(x2, y2) if olap2(p, q)]

# Outersection.  Segments intersect somewhere, not necessarily inside box
isct(((xp1, xp2), (yp1, yp2)), ((xq1, xq2), (yq1, yq2))) =
    [yp1-yp2 xp2-xp1; yq1-yq2 xq2-xq1] \ [yp1*xp2-yp2*xp1; yq1*xq2-yq2*xq1] |> Tuple
nullbox((x,y)) = ((x,x), (y,y))
intersections(x1, y1, x2, y2) =
    [(j, k, r)
        for (j, p, k, q) in nears(x1, y1, x2, y2)
        for r in [isct(p, q)]
        if olap2(p, nullbox(r)) && olap2(q, nullbox(r))
    ]
crossing(x1, y1, x2, y2, (j1, j2, (x,y))) =
    [x1[1:j1]; x; x2[j2+1:end]], [y1[1:j1]; y; y2[j2+1:end]]
    
# The lines must be on the same sides as in the call to intersections
cut((_, j, (x,y)), xx::Vector, yy::Vector) = [x; xx[j+1:end]], [y; yy[j+1:end]]
cut(xx::Vector, yy::Vector, (j, _, (x,y))) = [xx[1:j]; x], [yy[1:j]; y]
cut(A, xx, yy, B) = cut(A, cut(xx, yy, B)...)

# Hint: the solid lines start at bottom right, and end on the W axis

A = intersections(Wp, Ep.-E0, Wc, E1 .- E0)[1]
WA = A[3][1]
B = intersections(Wt, Et.-E0, Wp, Ep.-E0)[1]
WB = B[3][1]

stbl = crossing(Wt, Et.-E0, Wp, Ep.-E0, B)
stbl = crossing(stbl..., Wc, E1 .- E0, A)

C = intersections(Wp, rp/RTF, [WA, WA], [0.2,1.0]) |> only
D = intersections([WB, WB], [0.2,1.0], Wp, rp/RTF) |> only
E = intersections(Wt, rt/RTF, [WB, WB], [0.2,1.0]) |> only

P1 = plot(stbl, lw=4, lc=:yellow)
plot!(Wc, [0.0, 0.0], lc=:darkgray)
plot!(Wc, E1 .- E0, lc=S, style=:dot)
plot!(Ws, Es .- E0, lc=S)
plot!(Wc, E2 .- E0, lc=P, style=:dot)
plot!(Wp, Ep .- E0, lc=P)
plot!(Wc, E3 .- E0, lc=T, style=:dot)
plot!(Wt, Et .- E0, lc=T)
ylims!(-0.4, 0.1)
ylabel!("E")
savefig("../figs/resp210713a.pdf")

P2 = plot(cut(D, Wp, rp/RTF, C), lw=4, lc=:yellow)
plot!(cut(Wt, rt/RTF, E), lw=4, lc=:yellow)
plot!(Ws, rs/RTF, lc=S)
plot!(Wp, rp/RTF, lc=P)
plot!(Wt, rt/RTF, lc=T)
xlabel!("W")
ylabel!("r_v/R_TF")
savefig("../figs/resp210713b.pdf")

# P3 = plot(Ws, Js, lc=S, size=(300,200))
# plot!(Wp, Jp/2, lc=P)
# plot!(Wt, Jt/3, lc=T)
# ylabel!("J/n_v")
# 
# P4  = plot(Ws, Fs, lc=S, size=(300,200))
# plot!(Wp, Fp, lc=P)
# plot!(Wt, Ft, lc=T)

plot(P1, P2, link=:x, layout=(2,1), size=(300, 400))
savefig("../figs/resp210713c.pdf")