# Find a vortex slightly offset in th lab frame

source = open(@__FILE__) do f
    read(f, String)
end

C = 3000
N = 100
l = 20.0

include("../system.jl")
include("../figs.jl")

φ = ground_state(φ, 0, 1e-6)
μ = dot(L(φ), φ) |> real
r_TF = sqrt(μ)
rr = (0:20)*r_TF/20
nin = [sum(@. abs2(φ)*(r<rr[j])) for j = eachindex(rr)]

θ = 0.2
bp = Float64[]
decays = []
for rv = rr
    j = argmin(@. abs(z-rv))
    n = abs2(φ[j])
    ξ = 1/sqrt(2n*C)	# rmp-81-647
    f(rv) = @. φ*(z-rv)/sqrt(abs2(z-rv)+2(ξ^2))
    push!(decays, f(rv))
    push!(bp, sum(conj(f(rv*exp(1im*θ))).*f(rv) |> imag))
end

rr = rr/r_TF
PF = plot(rr, nin, label="Wu",
    leg=:none, framestyle=:box,
    fontfamily="Latin Modern Sans", ms=2,
    size=(200,200), dpi=72)
scatter!(rr, -bp./θ, ms=2, label="Imp. Berry")
xlims!(0,1)
ylims!(0,1)
savefig(PF, "../figs/resp200714a.pdf")
