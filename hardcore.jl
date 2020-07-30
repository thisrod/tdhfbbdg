# Berry phase with a cylinder core cut out

N = 400
l = 7
C = NaN
R = 2.0
rc = 1.0	# core radius

include("system.jl")
include("figs.jl")

hh = 2π*(0:0.01:1)
uu = exp.(1im*hh)

function imprint!(q, a)
    @. q *= (z-a)/abs(z-a)
    q[@. abs(z-a) < rc] .= 0
    q
end
imprint(a) = imprint!(ones(eltype(z), N,N), a)

lpsr = [rc*real(u)+sqrt(R^2-rc^2*imag(u)^2) for u in uu]
lapse = [2angle(u) for u in uu]
bp = pci([imprint(R*u) for u = uu])

# scatter(hh, h^2/π*bphase([imprint(R*u) for u = uu]), leg=:none)
# plot(pci([imprint(R*u) for u = uu]) |> sense_portrait |> implot, aspect_ratio=1)
# scatter(y, -slice(bp), ms=3, mc=:black, msw=0, leg=:none)
# plot!(lpsr[@. lapse ≥0], lapse[@. lapse ≥0], lc=:red)

θ = 0.3
r₁ = R*cos(θ) - sqrt(rc^2-R^2*sin(θ)^2)
w = r₁*exp(1im)
PA = zplot(imprint(R*exp(1im*(1+θ))))
plot!(R*cos.(hh), R*sin.(hh), lc=:white, leg=:none)
scatter!(real.([w]), imag.([w]), mc=:white)

PB = zplot(imprint(R*exp(1im*(1-θ))))
plot!(R*cos.(hh), R*sin.(hh), lc=:white, leg=:none)
scatter!(real.([w]), imag.([w]), mc=:white)
