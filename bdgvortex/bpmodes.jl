# Berry phase of rotating BdG modes
# BV* directories on ozstar

using LinearAlgebra, JLD2, Printf, Plots, PhasePlots
using Superfluids
using Superfluids: bdg_output, find_vortices
using Statistics: mean

include("../gpvortex/berry_utils.jl")

R = 10
# s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2) |> Superfluids.default!
# d = FourierDiscretisation{2}(100, 2R/99) |> Superfluids.default!
s = Superfluid{2}(100.0, (x,y)->(x^2+y^2)/2) |> Superfluids.default!
d = FourierDiscretisation{2}(100, 0.121) |> Superfluids.default!

Superfluids.default!(:xlims, (-5,5))
Superfluids.default!(:ylims, (-5,5))
Plots.default(:legend, :none)

J = Superfluids.operators(:J) |> only
Jev(u) = dot(u, J(u))/sum(abs2,u) |> real
Jev(u,v) =
    (norm2(u)*(Jev(u) - Jev(q)) - norm2(v)*(Jev(v) - Jev(q)))/(norm2(u)+norm2(v))

slice(u) = mean(u[:,(d.n÷2):(d.n÷2)+1], dims=2)

# r1 = 2.5;  Q7 = steady_state(rvs=[0; @. r1*exp(2π*im*(0:5)/6)], as=fill(0.1,7), Ω=0.6);
# plot(d, Q7, size=(200,200))
# plot!(r1*cos.(2π*(0:59)/50), r1*sin.(2π*(0:59)/50), lc=:white)
# savefig("../figs/resp210517a.pdf")

function loadmerge(ss...)
    global rr, Ωs, qs, ews, evs
    RR = [];  WS = [];  QS = [];  EWS = [];  EVS = []
    for s in ss
        @load s rr Ωs qs ews evs
        rr = rr[eachindex(Ωs)]  # jobs that timed out
        RR = [rr; RR]
        ixs = sortperm(RR)
        RR = RR[ixs]
        WS = [Ωs; WS];  WS = WS[ixs]
        QS = [qs; QS];  QS = QS[ixs]
        EWS = [ews; EWS];  EWS = EWS[ixs]
        EVS = [evs; EVS];  EVS = EVS[ixs]
    end
    rr, Ωs, qs, ews, evs = RR, WS, QS, EWS, EVS
    nothing
end

norm2(u) = sum(abs2, u)
area_enclosed(zs) = imag(sum(conj(zs[1:end-1]).*zs[2:end])/2)

function select(n)
    global Ω, q, rv, rvs, ws, us, vs, ja, jb, jc
    Ω = Ωs[n]
    q = qs[n]
    rv = mean(abs, find_vortices(d, q))
    rvs = find_vortices(d, q)
    
    ws, us, vs = bdg_output(d, ews[n], evs[n], a_tol=0.1)
    pushfirst!(ws,0)
    pushfirst!(us, q)
    pushfirst!(vs, q)
    ja = argmin(@. abs(ws-1+Ω))
    jb = argmin(@. abs(ws-2))
    jc = argmin(@. abs(ws-1-Ω))
    nothing
end

# loadmerge("firmpairmodes.jld2", "firmpairouter.jld2", "firmpairstable.jld2")
# ix = @. !isapprox(Ωs, 0.6, atol=1e-3);
# rr = rr[ix];  Ωs = Ωs[ix];  qs = qs[ix];  ews = ews[ix];  evs = evs[ix];

# @load "modes.jld2"
# rr1 = rr;
# RTF = rr[end];
# Lau1 = [(select(n); Jev(us[ja]))-Jev(q) for n in eachindex(rr1)];
# Lav1 = [(select(n); Jev(vs[ja]))-Jev(q) for n in eachindex(rr1)];
# Lcu1 = [(select(n); Jev(us[jc]))-Jev(q) for n in eachindex(rr1)];
# Lcv1 = [(select(n); Jev(vs[jc]))-Jev(q) for n in eachindex(rr1)];
# @load "pairmodes.jld2"
# rr2 = rr;
# Lau2 = [(select(n); Jev(us[ja]))-Jev(q) for n in eachindex(rr2)];
# Lav2 = [(select(n); Jev(vs[ja]))-Jev(q) for n in eachindex(rr2)];
# Lcu2 = [(select(n); Jev(us[jc]))-Jev(q) for n in eachindex(rr2)];
# Lcv2 = [(select(n); Jev(vs[jc]))-Jev(q) for n in eachindex(rr2)];
# @load "triplemodes.jld2"
# rr3 = rr[begin:end-1];
# Lau3 = [(select(n); Jev(us[ja]))-Jev(q) for n in eachindex(rr3)];
# Lav3 = [(select(n); Jev(vs[ja]))-Jev(q) for n in eachindex(rr3)];
# Lcu3 = [(select(n); Jev(us[jc]))-Jev(q) for n in eachindex(rr3)];
# Lcv3 = [(select(n); Jev(vs[jc]))-Jev(q) for n in eachindex(rr3)];
# Plots.default(:size, (300,200))
# plot(rr1/R_TF, Lbu1, legend=:topleft, label="1 u", size=(300,200))
# plot(rr1/RTF, Lau1, color=:red)
# plot!(rr1/RTF, Lav1, color=:red, style=:dot)

gpe_kelvon(j, ε) = q + ε*us[j] + conj(ε*vs[j])

match_order(rv0, rvs) = [rvs[argmin(@. abs(rvs-r))] for r in rv0]

function orbits(j, ε, rv0)
    hh = (0:20)/20*2π
    hcat([
        let rvs = find_vortices(d, gpe_kelvon(j, ε*exp(im*h)))
            match_order(rv0, rvs)
        end
        for h in hh]...) |> transpose
end

function animate_orb(j, ε)
    os = orbits(j, ε, rv0) |> collect
    for i = 1:20
        bc = pci([gpe_kelvon(j, ε*exp(im*h)) for h = (0:i)/20*2π])
        plot(d, bc, aspect_ratio=1, xlims=(-2.5,2.5), ylims=(-2.5,2.5))
        plot!(real(os), imag(os), lc=:black)
        scatter!(real(os[i,:]), imag(os[i,:]), mc=:black, ms=3, msw=0)
        display(current())
        sleep(0.5)
    end
end

function orbcrv(j, n; kwargs...)
    ε = 0.2
    bc = pci([gpe_kelvon(j, ε*exp(im*h)) for h = (0:20)/20*2π])
    plot(d, bc, aspect_ratio=1; kwargs...)
    for e = (1:4)/10
        e ≈ ε && continue
        os = orbits(j, e, rv0) |> collect
        plot!(real(os), imag(os), lc=:lightgray)
    end
    os = orbits(j, ε, rv0) |> collect
    plot!(real(os), imag(os), lc=:black)
    scatter!(real(os[1,:]), imag(os[1,:]), mc=:black, ms=3, msw=0)
    plot!(real(os[1:n,:]), imag(os[1:n,:]), lc=:black, arrow=true)
    xlims!(-2.5,2.5)
    ylims!(-2.5,2.5)
    current()
end

Q = steady_state()

function phases(ε, j)
    z = argand(d)
    nv = mean(abs2.(Q[[argmin(@. abs(z-r)) for r in rv0]])) / d.h^2
    ain = [area_enclosed(o) for o in eachcol(orbits(j, ε, rv0) .- transpose(rv0))] |> sum
    bp = bphase([gpe_kelvon(j, ε*exp(im*h)) for h = (0:20)/20*2π])[end]
    -bp/2π, nv*ain
end

# j = 2
# εs = [range(0.005, 0.1, length=9); 0.05*(2:10)];
# scatter(εs, [phases(ε, j)[i] for ε in εs, i in 1:2])
# scatter(εs, [/(x...) for x in phases.(εs, j)])

# @load "pairmodes.jld2"
# select(3);  rv0 = find_vortices(d,q)
# plot(core.([Q, us[2:3]..., q, vs[2:3]..., Q])..., orbcrv(2, 5), orbcrv(3,4), layout=(3,3), size=(600,600))
# savefig("../figs/resp210426a.pdf")

# @load "triplemodes.jld2"
# select(3);  rv0 = find_vortices(d,q);  plot(core.([Q, us[2:4]..., q, vs[2:4]..., Q])..., orbcrv(2,5), orbcrv(3,5), orbcrv(4,5), layout=(3,4), size=(800,600))

# 
# Plots of order parameters and sound wave modes
#

core!() = scatter!(real(rvs), imag(rvs), mc=:black, ms=2, msw=0)
core(u) = (plot(d,u);  core!())
att!(key::Symbol, s::String, color=:white) = att!(Val(key), s, color)
att!(key::Symbol, j::Integer, color=:white) = att!(Val(key), (@sprintf "%d" j), color)
att!(key::Symbol, x::Real, color=:white) = att!(Val(key), (@sprintf "%.3f" x), color)
att!(::Val{:mode}, s, color) = annotate!(5.7,5.7,text(s,:top,:right, 7, color))

function slt(u1=us[1], u2=us[2])	# split frame
    z = argand(d)
    s = similar(q)
    ixs = @. imag(z) > 0
    s[ixs] = u1[ixs] / maximum(abs, u1)
    ixs = @. imag(z) < 0
    s[ixs] = u2[ixs] / maximum(abs, u2)
    s
end

# x, _ = Superfluids.daxes(d)
# p(q; kwargs...) = (phaseplot(x, x, q, γ=1.7, xlims=(-5,5), ylims=(-5,5), size=(200,200); kwargs...); core!())
# p(q, r) = (clim = maximum(abs, [q r]); (p(q; clim), p(r; clim)))
# Plots.default(:frame, :box)
# ax = -4:2:4; ax = (ax, string.(ax))

# @load "firmmodes.jld2"
# pts = []
# select(1); push!(pts, p(slt(), xticks=ax, yticks=ax));  savefig("../figs/resp210818a.pdf")
# select(3); push!(pts, p(slt(), xticks=ax, yticks=((),())));  savefig("../figs/resp210818b.pdf")
# select(6); push!(pts, p(slt(), xticks=ax, yticks=((),())));  savefig("../figs/resp210818c.pdf")
# select(8); push!(pts, p(slt(), xticks=ax, yticks=((),())));  savefig("../figs/resp210818d.pdf")
# push!(pts, p(slt((@.us[1]/abs(us[1])), (@.us[2]/abs(us[2]))), xticks=ax, yticks=((),())));  savefig("../figs/resp210818e.pdf")
# plot(pts..., layout=(1,5), size=(1000,200));  savefig("../figs/resp210818f.pdf")

# @load "firmpairmodes.jld2"
# pts = []
# select(4); push!(pts, p(slt(), xticks=ax, yticks=ax));  savefig("../figs/resp210819a.pdf")
# @load "firmtriplemodes.jld2"
# select(4); push!(pts, p(slt(), xticks=ax, yticks=((),())));  savefig("../figs/resp210819b.pdf")
# @load "sevenmodes.jld2"
# select(1); push!(pts, p(slt(), xticks=ax, yticks=((),())));  savefig("../figs/resp210819c.pdf")
# plot(pts..., layout=(1,3), size=(600,200));  savefig("../figs/resp210819d.pdf")



# nk = fill(3, length(Ωs));  nk[1] = 5;

# nl = [5, 4, 4]

plot_oprm() =
    plot(core!(q, @sprintf "W = %.3f" Ω), core!(@. q/abs(q)), size=(400,200))
    
function plot_spectrum(j=nothing)
    uns = norm2.(us)
    vns = norm2.(vs)
    ns = vns./(uns .- vns)
    # handle zero modes with norm -bignum
    ns[@. abs(ns) > 1e6] .= Inf
    ms = 5sqrt.(ns)
    mc = [s > 25 ? :green : :black for s in ms]
    ms[ms .< 2] .= 2
    ms[ms .> 25] .= 5
    Js = Jev.(us, vs)
    jj = [minimum(Js), 0, maximum(Js)]
    ww = abs.(jj)
    plot(jj, ww, lc=:gray)
    plot!(jj, ww .+ 2, lc=:gray)
    scatter!(Js, real(ws) + Ω*Js; ms, mc, msw=0)
    isnothing(j) || scatter!([Js[j]], [real(ws[j])+ Ω*Js[j]], ms=ms[j], mc=:red, msw=0)
end

function plot_mode(j)
    P = core(us[j])
    att!(:mode, j)
    plot(plot_spectrum(j), P, core.([vs[j], (@. us[j]/abs(us[j])), (@. vs[j]/abs(vs[j]))])...,
        size=(400,600), layout=@layout[a; b c; d e])
end

animate_spectrum() =
    @animate for n = eachindex(Ωs)
        select(n)
        plot_spectrum()
        xlabel!("J")
        ylabel!("w")
        title!(@sprintf "rv = %.1f" rv)
    end

# kw = [ew[1] for ew in ews]

function plot_kelvon(n)
    select(n)
    P = scatter(rr/rr[end], real(kw), mc=:black, msw=0)
    scatter!(rr/rr[end], abs.(imag(kw)), mc=:red, msw=0)
    plot!(fill(rr[n]/rr[end], 2), collect(ylims()))
    plot(P, core!.([us[2], vs[2], (@. us[2]/abs(us[2])), (@. vs[2]/abs(vs[2]))])...,
        size=(400,600), layout=@layout[a; b c; d e])
end

function zeroness(u, v)
   u = normalize(u)
   v = normalize(v)
   _, R = qr([u[:] v[:]])
   abs(R[2,2]), R[1,2]/abs(R[1,2])
end

function plot_pci(j, urs=rot(us[j]), vrs=rot(vs[j]))
    nin = sum(abs2, q[@. abs(z) < rv])
    nout = sum(abs2, q[@. abs(z) > rv])
    f(us) =
        core!(pci(us), j, bphase(us)[end]/2π, bphase(us)[end]/2π/nin, bphase(us)[end]/2π/nout, color=:black)
    plot(f(urs), f(vrs), size=(400,200))
end

animate_modes() =
    for j = sortperm(ws + Ω*Jev.(us, vs))
        plot_mode(j) |> display
        sleep(1.5)
    end

# 
# Plot the angular momenta of the sound wave functions, which
# determine their Berry phases.
# 

mode_colors = [:black, :red, :green, :cyan, :magenta]

function Ls(js...)
    sel = isempty(js)
    sel && (js = 1:3)
    Lus = fill(NaN, length(rr), length(js))
    Lvs = copy(Lus)
    Ltot = copy(Lus)
    for n in eachindex(rr)
        select(n)
        sel && (js = [ja, jb, jc])
        Lus[n,:] = [Jev(us[j]) for j in js] .- Jev(q)
        Lvs[n,:] = -[Jev(vs[j]) for j in js] .+ Jev(q)
        Ltot[n,:] = [Jev(us[j], vs[j]) for j in js]
    end
    Lus, Lvs, Ltot
end

# Lus, _, _ = Ls(2)
# P1 = plot(rr/RTF, Lus, lc=:black, size=(200,200))
# xlabel!("r_v/R_TF")
# ylabel!("J_k")
# Lus, Lvs, Ltot = Ls()
# P2 = plot(rr/RTF, Ltot, color_palette=[:green, :red, :blue], size=(200,200))
# plot!(rr/RTF, Lus, ls=:dashdot, color_palette=[:green, :red, :blue])
# plot!(rr/RTF, Lvs, ls=:dashdotdot, color_palette=[:green, :red, :blue])
# xlabel!("r_v/R_TF")
# ylabel!("J")
# plot(P1, P2, size=(400, 200))
# savefig("../figs/resp210630a.pdf")

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

function animate_pair_qdb()
    @animate for n = eachindex(rr)
        select(n)
        plot([plot(d, x)
            for j in 1:3
            for x in Any[us[j], pcius[n, j], vs[j], pcivs[n,j]]
            ]...,
            size=(4*200,3*200),
            layout=(3,4)
        )
    end
end

function plot_bps()
    Lus = fill(NaN, length(rr), 5)
    Lvs = fill(NaN, length(rr), 5)
    for n = eachindex(rr)
        select(n)
        Lus[n,:] = [Jev(us[j]) for j in [1, 2, ja, jb, jc]]
        Lvs[n,:] = [Jev(vs[j]) for j in [1, 2, ja, jb, jc]]
    end
    plot(rr, [bpus -bpvs]/2π, color_palette=mode_colors, ls=:dot)
    plot!(rr, [Lus -Lvs], color_palette=mode_colors, ls=:dash)
    plot!(rr, bpod/2π, ls=:dash, lc=:orange)
    xlabel!("rv")
    title!("BP/2pi n (dashes) vs J (dots) for u and v")
end

pair_colors = [:black, :green, :blue]

function plot_pair_bps()
    Lus = fill(NaN, length(rr), 3)
    Lvs = fill(NaN, length(rr), 3)
    for n = eachindex(rr)
        select(n)
        Lus[n,:] = Jev.(us[1:3])
        Lvs[n,:] = Jev.(vs[1:3])
    end
    plot(rr, [bpus -bpvs]/2π, color_palette=pair_colors, ls=:dot)
    plot!(rr, [Lus -Lvs], color_palette=pair_colors, ls=:dash)
    xlabel!("rv")
    title!("BP/2pi n (dashes) vs J (dots) for u and v")
end

animate_ods() =
    @animate for n = eachindex(rr)
        select(n)
        P = scatter(rr, bpod/2π)
        scatter!([rr[n]], [bpod[n]/2π], mc=:red)
        xlabel!("rv")
        ylabel!("BP/2pi")
        plot(P, plot(d, @. us[1]/abs(us[1])), plot(d, pciod[n]), plot(d, @. us[2]/abs(us[2])), layout=(2,2), size=(400,400))
    end
