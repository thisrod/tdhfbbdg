# Vortices and solitons

N = 400
l = 15
C = NaN
R = 2.0
rc = 1.0	# core radius

include("system.jl")
include("figs.jl")

hh = 2π*(0:0.01:1)
uu = exp.(1im*hh)

function imprint!(q, a, anti=false)
    @. q *= anti ? conj(z-a) : (z-a)
    @. q /= √(1+abs2(z-a))
end

imprint(q, args...) = imprint!(copy(q), args...)

function offtest(a, θ=0)
    v = z.-a
    @. v /= abs(v)
    v[@. abs(z-a-0.5unit(a)*exp(1im*θ)) < 1] .= 0
    v
end

function vs(s, a=0, θ=angle(a); core=:soft)
    w = @. exp(-1im*θ)*(z-a)
    v = s*w+(1-s)*conj(w)
    @. v *= exp(1im*sign(2s-1)θ)
    if core == :soft
        @. v /= √(1+abs2(v))
    elseif core == :hard
        v[@. abs(v)<1] .= 0
        v[@. abs(v) != 0] ./= abs.(v[@. abs(v) != 0])
    elseif core == :cylinder
        @. v /= abs(v)
        v[@. abs(z-a)<1] .= 0
    elseif core == :none
        @. v /= abs(v)
    elseif isreal(core)
        @. v /= abs(v)
        rot = @. (z-a)/unit(a)
        ix = @. hypot(core*real(rot), 1/core*imag(rot)) < 1
        v[ix] .= 0
    else
        error("core = $(core) not supported")
    end
    v
end

# rr = [0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.7, 1.0]
# for j = eachindex(rr)
#     twoplot(vs(rr[j]))
#     savefig("figs/resp200718$('a'+j-1).pdf")
# end

"Approximate a soliton as a string of vortices"
function roat(m,ξ=0.15)
    c = exp(2π*1im/m)
    u = similar(z)
    fill!(u,1)
    for j = 1:2:m
        v = z.-2*c^j
        @. u *= v/√(ξ^2+abs2(v))
    end
    for j = 2:2:m
        v = conj(z).-2*c^j
        @. u *= v/√(ξ^2+abs2(v))
    end
    u
end

# mm = [4, 10, 18, 30, 60]
# for j = eachindex(mm)
#     twoplot(roat(mm[j]))
#     savefig("figs/resp200718$('a'+j+length(rr)-1).pdf")
# end

# PA = plot()
# bps = Float64[]
# for s = 0:0.3:2π
#     bp = pci([offtest(R*u, s) for u = uu])
#     push!(bps, h^2*sum(bp) / (2π^2*R^2))
# end

# bp = pci([vs(0.2, R*u) for u = uu])
# h^2*sum(bp) / (2π^2*R^2)
# plot(bp |> sense_portrait |> implot, aspect_ratio=1)
#     scatter!(PA, hh, h^2/π*bphase([vs(s, u; core=false).*(r.<N*h/2) for u = uu]), label="$s", leg=:topleft)
