# plotting functions

using Plots, ComplexPhasePortrait

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

show_vortex!(u) = show_vortex!(u, :white)

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

pci(S) = [@. imag(conj(S[j+1])*S[j]) for j = 1:length(S)-1] |> sum
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
