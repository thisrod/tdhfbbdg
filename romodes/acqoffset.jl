# Find a vortex slightly offset in th lab frame

source = open(@__FILE__) do f
    read(f, String)
end

using Arpack, JLD2

C = 3000
N = 100
l = 20.0

include("../system.jl")
include("../figs.jl")

r = abs.(z)

φ = ground_state(φ, 0, 1e-6)
μ = dot(L(φ), φ) |> real
r_TF = sqrt(μ)
rr = 0:h:r_TF
nin = [sum(@. abs2(φ)*(r<rr[j])) for j = eachindex(rr)]

@. φ *= (z-0.2)
φ /= norm(φ)
φ = ground_state(φ, 0, 1e-1)

totreps = 5300
decays = [φ]
for j = 1:5:20
    a = totreps^2/20
    reps = ceil(Int, sqrt(a*(j+1)) - sqrt(a*j))
    result = optimize(
        ψ -> dot(ψ,H(ψ)) |> real,
        (buf,ψ) -> copyto!(buf, 2*L(ψ)),
        decays[end],
        GradientDescent(manifold=Sphere()),
        Optim.Options(iterations=3000, g_tol=1e-9, allow_f_increases=true)
    )
    push!(decays, result.minimizer)
end

function vphase(q)
    w = find_vortex(q)
    @. abs(q)*exp(1im*angle(z-w))
end

rc = @. abs(find_vortex(decays))
bp1 = Float64[]
bp2 = Float64[]

θ = 0.2
for j = eachindex(decays)
    u = decays[j]
    v = u-1im*θ*J(u)-θ^2/2*J(J(u))+1im*θ^3/factorial(3)*J(J(J(u)))
    push!(bp1, sum(conj.(v).*u |> imag))
    push!(bp2, sum(conj.(vphase(v)).*vphase(u) |> imag))
end

function diagnostic(u)
    v = u-1im*θ*J(u)-θ^2/2*J(J(u))+1im*θ^3/factorial(3)*J(J(J(u)))
    plot(zplot(u), zplot(v), zplot(vphase(u)), zplot(vphase(v)), xlims=(-5,5), ylims=(-5,5))
end

rc /= r_TF
rr = rr/r_TF
PF = plot(rr, nin, label="Wu",
    leg=:none, framestyle=:box,
    fontfamily="Latin Modern Sans", ms=2,
    size=(200,200), dpi=72)
scatter!(rc, -bp2./θ, ms=2, label="Imp. Berry")
xlims!(0,1)
ylims!(0,1)
savefig(PF, "../figs/resp200713a.pdf")
