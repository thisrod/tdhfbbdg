# Vortices and solitons

N = 300
l = 7
C = NaN

include("system.jl")
include("figs.jl")

function imprint!(q, a, anti=true)
    @. q *= anti ? conj(z-q) : (z-a)
    @. q /= √(1+abs2(z-a))
end

imprint(q, args...) = imprint!(copy(q), args...)

function vs(r)
    v = r*z+(1-r)*conj(z)
    @. v /= √(1+abs2(v))
end

# rr = [0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.7, 1.0]
# for j = eachindex(rr)
#     twoplot(vs(rr[j]))
#     savefig("figs/resp200718$('a'+j-1).pdf")
# end

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
