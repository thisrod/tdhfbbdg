# Check grid and residual convergence

using LinearAlgebra, JLD2, Printf, Plots
using Superfluids
using Superfluids: bdg_output

include("../gpvortex/berry_utils.jl")

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2) |> Superfluids.default!
d = FourierDiscretisation{2}(100, 20/99) |> Superfluids.default!
d2 = FourierDiscretisation{2}(80, 20/79)

R = 10
z = argand(d)
z_in = z[@. abs(z) < R]

function rotated(q, θ)
    u = zero(z)
    u[@. abs(z) < R] = [interpolate(d, q, w*exp(-1im*θ)) for w in z_in]
    u
end

hh = (0:40)*2π/40
rot(u::Array) = [rotated(normalize(u),h) for h in hh]

J = Superfluids.operators(:J) |> only
Jev(u) = dot(u, J(u))/sum(abs2,u) |> real
Jev(u,v) =
    (norm2(u)*(Jev(u) - Jev(q)) - norm2(v)*(Jev(v) - Jev(q)))/(norm2(u)+norm2(v))

U = Array{Any}(undef, 5, 3)
V = similar(U)
ωs = similar(U)

function extract_modes!(d, F, i)
    local ws, us, vs
    for (n, ew, ev) = zip(1:5, F["ews"], F["evs"])
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
        @assert @isdefined us
        ωs[n, i] = ws
        U[n, i] = us
        V[n, i] = vs
    end
end

jldopen("out.jld2", "r") do F
    global rr = F["rr"]
    global Ωs = collect(Float64, F["ws"])
    global qs = F["qs"]
    extract_modes!(d, F, 1)
end
jldopen("outrelax.jld2", "r") do F
    @assert F["rr"][10:-1:6] ≈ rr
    global Ωs = [Ωs F["ws"]]
    global qs = [qs F["qs"]]
    extract_modes!(d, F, 2)
end
jldopen("outcoarse.jld2", "r") do F
    # grid changes R_TF thus rr
    resample(u) = [interpolate(d2, u, r) for r in argand(d)]
    @assert isapprox(F["rr"][10:-1:6], rr, atol=1e-6)
    global Ωs = [Ωs F["ws"]]
    global qs = [qs @. normalize(resample(F["qs"]))]
    extract_modes!(d2, F, 3)
    for n = 1:5
        U[n,3] = resample.(U[n,3])
        V[n,3] = resample.(V[n,3])
    end
end

# Compare to the high res, high relaxed version

# scatter(rr, Ωs .- Ωs[:,2], legend=true)
# @. norm(qs[:,[1,3]] - qs[:,2])
# plot([plot(d,q) for q in qs[:,[1,3]] .- qs[:,2]]..., layout=(2,5), size=(1000, 400))
# @. norm(qs[:,[1,3]] - qs[:,2])
# @. norm(ωs[:,[1,3]] - ωs[:,2], Inf)
# scatter(ipangle.(U[1,1], U[1,2]))
# norm.([ipangle.(U[n,i], U[n,2]) for n=1:5, i=[1,3]], Inf)
# let ua = U[5,3][1], ub = U[5,2][1]; plot([plot(d, u) for u in Any[ua, (@. ua/abs(ua)), pci(rua), ub, (@. ub/abs(ub)), pci(rub)]]..., layout=(2,3), size=(600,400)) end

norm2(u) = sum(abs2, u)
ipangle(u, v) = acos(dot(u, v)/norm(u)/norm(v) |> abs)


# core!() = scatter!([rv], [0], mc=:white, ms=2, msw=1)
# core!(u) = (plot(d,u);  core!())
# core!(u,s::String; color=:white) = (core!(u); annotate!(5.7,5.7,text(s,:top,:right, 7, color)))
# core!(u,j::Int; color=:white) = (core!(u); annotate!(-5.7,5.7,text((@sprintf "%d" j),:top,:left, 7, color)))
# function core!(u,j::Int, bs...; color=:white)
#     core!(u)
#     annotate!(-5.7,5.7,text((@sprintf "%d" j),:top,:left, 7, color))
#     for (i, b) in pairs(bs)
#         isnan(b) || isinf(b) || annotate!(5.7,5.7-0.7i,text((@sprintf "%.3f" b),:top,:right, 7, color))
#     end
#     current()
# end

plot_oprm() =
    plot(core!(q, @sprintf "W = %.3f" Ω), core!(@. q/abs(q)), size=(400,200))
    
function plot_spectrum(n, j)
    us = U[n,j];  vs = V[n,j];  ws = ωs[n,j]
    uns = [sum(abs2, u) for u in us[3:end]]
    vns = [sum(abs2, v) for v in vs[3:end]]
    ns = sqrt.(vns./(uns .- vns))
    scatter(Jev.(us, vs), ws, ms=max.(5*[1; 1; ns], 2), mc=:black, msw=0)
end

animate_spectrum() =
    @animate for n = eachindex(Ωs)
        select(n)
        plot_spectrum()
        xlabel!("J")
        ylabel!("w")
        title!(@sprintf "rv = %.1f" rv)
    end

function plot_mode(j)
    P = plot_spectrum()
    scatter!([Jev(us[j], vs[j])], [ws[j]], mc=:red)
    plot(P, core!(us[j], j), core!.([vs[j], (@. us[j]/abs(us[j])), (@. vs[j]/abs(vs[j]))])...,
        size=(400,600), layout=@layout[a; b c; d e])
end

function plot_pci(j, urs=rot(us[j]), vrs=rot(vs[j]))
    nin = sum(abs2, q[@. abs(z) < rv])
    nout = sum(abs2, q[@. abs(z) > rv])
    f(us) =
        core!(pci(us), j, bphase(us)[end]/2π, bphase(us)[end]/2π/nin, bphase(us)[end]/2π/nout, color=:black)
    plot(f(urs), f(vrs), size=(400,200))
end

animate_modes() =
    for j = eachindex(ws)
        plot_mode(j) |> display
        sleep(1.5)
    end
    
mode_colors = [:black, :red, :green, :cyan, :magenta]

function plot_amom()
    Lq = fill(NaN, length(rr))
    Lus = fill(NaN, length(rr), 5)
    Lvs = fill(NaN, length(rr), 5)
    Ltot = fill(NaN, length(rr), 5)
    for n = eachindex(rr)
        select(n)
        Lq[n] = Jev(q)
        Lus[n,:] = [Jev(us[j]) for j in [1, 2, ja, jb, jc]]
        Lvs[n,:] = [Jev(vs[j]) for j in [1, 2, ja, jb, jc]]
        Ltot[n,:] = [Jev(us[j], vs[j]) for j in [1, 2, ja, jb, jc]]
    end
    plot(rr, Lus.-Lq, ls=:dashdot, color_palette=mode_colors)
    plot!(rr, -Lvs.+Lq, ls=:dashdotdot)
    plot!(rr, Ltot, ls=:solid)
    xlabel!("rv")
    ylabel!("L-Loprm")
    title!("C, K, Dp, Dm, B.  u 1 dot, -v 2 dots")
    ylims!(-1.5,2)
end

# TODO refer BdG modes to oprm phase

function animate_qdb()
    @animate for n = eachindex(rr)
        select(n)
        plot([core!(x)
            for (i, j) in pairs([1, 2, ja, jb, jc])
            for x in Any[us[j], pcius[n, i], vs[j], pcivs[n, i]]
            ]...,
            size=(800,1000),
            layout=(5,4)
        )
    end
end

function plot_bps(ns=ones(length(rr)))
    Lus = fill(NaN, length(rr), 5)
    Lvs = fill(NaN, length(rr), 5)
    for n = eachindex(rr)
        select(n)
        Lus[n,:] = [Jev(us[j]) for j in [1, 2, ja, jb, jc]]
        Lvs[n,:] = [Jev(vs[j]) for j in [1, 2, ja, jb, jc]]
    end
    plot(rr, [bpus -bpvs]/2π, color_palette=mode_colors, ls=:dot)
    plot!(rr, [Lus -Lvs], color_palette=mode_colors, ls=:dash)
end
