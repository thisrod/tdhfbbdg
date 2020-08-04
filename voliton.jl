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

function vs(r, a=0, θ=angle(a); core=true)
    w = @. exp(-1im*θ)*(z-a)
    v = r*w+(1-r)*conj(w)
    @. v *= exp(1im*sign(2r-1)θ)
    @. v /= core ? √(1+abs2(v)) : abs(v)
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

PA = plot()
for s = [0, 0.45, 0.47, 0.5, 0.53, 0.55, 1]
    scatter!(PA, hh, h^2/π*bphase([vs(s, u; core=false).*(r.<N*h/2) for u = uu]), label="$s", leg=:topleft)
end

bp = pci([vs(0.2, R*u) for u = uu])
# h^2*sum(bp) / (2π^2*R^2)
# plot(bp |> sense_portrait |> implot, aspect_ratio=1)