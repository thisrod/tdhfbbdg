# plotting functions

using Plots, ComplexPhasePortrait, Printf

function poles(u)
    rs = (-1:1)' .+ 1im*(-1:1)
    rs *= h
    conv(u, rs) = [rs.*u[j:j+2,k:k+2] |> sum for j = 1:N-2, k = 1:N-2]
    P = zero(u)
    P[2:end-1,2:end-1] .= conv(u, conj(rs))
    Q = zero(u)
    Q[2:end-1,2:end-1] .= conv(u, rs)
    P, Q
end

function find_vortex(u)
    w = poles(u) |> first .|> real
    z[argmax(w)]
end

function show_vortex!(u, clr)
    v0 = find_vortex(u)
    scatter!([real(v0)],[imag(v0)],m=:circle, ms=1, mc=clr, msw=0, leg=:none)
end

show_vortex!(u) = show_vortex!(u, :white)

function bphase(S, ts)
    bp = [dot(S[j+1], S[j]) |> imag for j = 1:length(S)-1] |> cumsum
    bp = [0; bp]
    zs = [find_vortex(S[j]) for j = eachindex(S)]
    nin = sum(abs2.(S[1][r .< mean(abs.(zs))]))
    P = scatter(ts, bp, label="Berry", leg=:topleft)
    scatter!(ts, nin*unroll(angle.(zs).-angle(zs[1])), label="Wu & Haldane")
    xlabel!("t")
    ylabel!("phi")
    title!("Geometric phases by GPE")
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
implot(x,y,image) = plot(x, y, image, yflip=false, aspect_ratio=1)
implot(image) = implot(y,y,image)
saneportrait(u) = reverse(portrait(u), dims=1)
zplot(u) = implot(saneportrait(u).*abs2.(u)/maximum(abs2,u))
argplot(u) = implot(saneportrait(u))
zplot(u::Matrix{<:Real}) = zplot(u .|> Complex)
argplot(u::Matrix{<:Real}) = argplot(u .|> Complex)
