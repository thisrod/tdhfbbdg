# plotting functions

using Plots, ComplexPhasePortrait, Colors

# Colors for Berry phase plots
snapcol = RGB(0.3,0,0)
bpsty, impsty, insty, outsty, nsty =
    distinguishable_colors(5+3, [snapcol, RGB(1,1,1), RGB(0,0,0)])[4:end]
bpsty = (ms=2, mc=bpsty, msc=0.5bpsty)
impsty = (ms=2, mc=impsty, msc=0.5impsty)
insty = (lc=insty,)
outsty = (lc=outsty,)
nsty = (lc=nsty,)

popts = (dpi=72, leg=:none, framestyle=:box, fontfamily="Latin Modern Sans")
imopts = (popts..., xlims=(-5,5), ylims=(-5,5), size=(200,200))
sqopts = (popts..., size=(200,200))
recopts = (popts..., size=(100,200))


"ComplexPhasePortrait, but with real sign instead of phase"
function sense_portrait(xs)
    mag = maximum(abs, xs)
    C = cgrad([:cyan, :white, :red])
    # TODO stability at |x| ≈ mag
    xs .|> (x -> C[(x+mag)/2mag])
end

function show_vortex!(u, clr=:white)
    v0 = find_vortex(u)
    scatter!([real(v0)],[imag(v0)],m=:circle, ms=2, mc=clr, msw=0, leg=:none)
end

function show_moat!(u, clr=:white)
    v0 = find_moat(u)
    scatter!([real(v0)],[imag(v0)],m=:circle, ms=2, mc=clr, msw=0, leg=:none)
end

function show_vortices!(u, clr=:white)
    v0 = find_vortex(u)
    v1 = find_moat(u)
    scatter!([real(v0)],[imag(v0)],m=:circle, ms=2, mc=clr, msw=0, leg=:none)
    scatter!([real(v1)],[imag(v1)],m=:circle, ms=2, mc=clr, msw=0, leg=:none)
end

# function bphase(S, ts)
#     bp = [dot(S[j+1], S[j]) |> imag for j = 1:length(S)-1] |> cumsum
#     bp = [0; bp]
#     zs = [find_vortex(S[j]) for j = eachindex(S)]
#     nin = sum(abs2.(S[1][r .< mean(abs.(zs))]))
#     P = scatter(ts, bp, label="Berry", leg=:topleft)
#     scatter!(ts, nin*unroll(angle.(zs).-angle(zs[1])), label="Wu & Haldane")
#     xlabel!("t")
#     ylabel!("phi")
#     title!("Geometric phases by GPE")
# end

"pci([q1, q2, ...]) pointwise Berry phase after sequence of states"
pci(S) = [@. imag(conj(S[j+1])*S[j]) for j = 1:length(S)-1] |> sum

"bphase([q1, q2, ...]) cumulative Berry phase for sequence of states"
function bphase(S)
    bp = [dot(S[j+1], S[j]) |> imag for j = 1:length(S)-1] |> cumsum
    [0; bp]
end

function unroll(θ)
    # reverse modulo 2π
    Θ = similar(θ)
    Θ[1] = θ[1]
    poff = 0.0
    for j = 2:length(θ)
        jump = θ[j] - θ[j-1]
        if jump > π
            poff -= 2π
        elseif jump < -π
            poff += 2π
        end
        Θ[j] = θ[j] + poff
    end
    Θ
end

"Wirtinger derivatives"
function widir(u, zz=[]; ms=0.5, rel=false, rad=5, mask=ones(size(u)))
    P, Q = poles(u)
    P .*= mask
    Q .*= mask
    plots = []
    vv = [P, (P+conj(Q))/2, (P-conj(Q))/2im, Q]
    ss = ["z", "abs", "arg", "z*"]
    for j = 1:4
        Pj = zplot(rel ? (@. vv[j]*(r<rad)/abs(u)) : vv[j])
        scatter!(real.(zz), imag.(zz), mc=:white, ms=ms, leg=:none)
        xlims!(-5,5)
        ylims!(-5,5)
        title!(ss[j])
        push!(plots, Pj)
    end
    plot(plots...)
end

# Plots and portraits are a bit wierd with pixel arrays
implot(x,y,image) = plot(x, y, image,
    yflip=false, aspect_ratio=1, framestyle=:box, tick_direction=:out)
implot(image) = implot(y,y,image)
saneportrait(u) = reverse(portrait(u), dims=1)
zplot(u) = implot(saneportrait(u).*abs2.(u)/maximum(abs2,u))
argplot(u) = implot(saneportrait(u))
zplot(u::Matrix{<:Real}) = zplot(u .|> Complex)
argplot(u::Matrix{<:Real}) = argplot(u .|> Complex)
twoplot(u) = plot(zplot(u), argplot(u))

"Scatter plot with phase as color"
function zscatter!(P, x, z; args...)
    pxl = similar(z, 1, 1)
    for j = eachindex(x)
        pxl .= z[j]
        scatter!(P, x[j:j], abs.(pxl), mc=portrait(pxl)[], msw=0, leg=:none; args...)
    end
    P
end
zscatter(x, z; args...) = zscatter!(plot(), x, z; args...)

# rasters in the moat, around θ
function waster(u, θ)
    P, Q = poles(u)
    ab = (P+conj(Q))/2
    ar = (P-conj(Q))/2im
    hh = @. angle(z*exp(-1im*θ))
    mix = @. (R-0.5w < r < R+0.5w) & (-0.3 < hh < 0.3)
    B = box(keys(u)[mix])
#    zscatter(hh[mix], ab[mix]), zscatter(hh[mix], ar[mix]),
#        zscatter(hh[mix], solrat(u)[mix]), 
    zplot((@. ab/abs(u))[B]), zplot((@. ar/abs(u))[B]),
        argplot(u[B]), zplot((solrat(u).*mix)[B])
end

# soliton ratio
function solrat(u)
    P, Q = poles(u)
    P = P[2:end-1,2:end-1]
    Q = Q[2:end-1,2:end-1]
    v = zero(u)
    v[2:end-1,2:end-1] = 1im*(P+conj(Q))./(P-conj(Q))
    v
end

function Plots.scatter!(zz::Vector{Complex}, args...)
    scatter!(real.(zz), imag.(zz), args)
end

function Plots.plot!(zz::Vector{Complex}, args...)
    plot!(real.(zz), imag.(zz), args)
end