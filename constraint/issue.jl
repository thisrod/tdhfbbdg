# relax an order parameter with a vortex position constraint

using LinearAlgebra, Plots, ComplexPhasePortrait, BandedMatrices, Optim

default(:legend, :none)

C = 3000
N = 100
h = 0.19801980198019803
x = h/2*(1-N:2:N-1);  y = x';  z = Complex.(x,y)
r = abs.(z)
V = abs2.(z)

function op(stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end
∂ = (1/h).*op(Float64[-1/2, 0, 1/2])
∂² = (1/h^2).*op(Float64[1, -2, 1])

T(ψ) = -(∂²*ψ+ψ*∂²')/2
U(ψ) = C/h*abs2.(ψ)
J(ψ) = -1im*(x.*(ψ*∂')-y.*(∂*ψ))
L(ψ,Ω) = L(ψ)-Ω*J(ψ)
L(ψ) = T(ψ)+(V+U(ψ)).*ψ
H(ψ,Ω) = H(ψ)-Ω*J(ψ)
H(ψ) = T(ψ)+(V+U(ψ)/2).*ψ

struct NormVort <: Manifold
   ixs
   o
   function NormVort(rv::Number)
        # pick 4 points nearest vortex
        ixs = sort(eachindex(z), by=j->abs(z[j]-rv))
        ixs = ixs[1:4]
        a = normalize!(z[ixs].-rv)
        o = ones(eltype(z), 4)
        o .-= a*(a'*o)
        normalize!(o)
        new(ixs, o)
   end
end

function project!(M, q)
    q[M.ixs] .-= M.o*(M.o'*q[M.ixs])
    q
end

# The "vortex at rv" subspace is invariant under normalisation
Optim.retract!(M::NormVort, q) =
    Optim.retract!(Sphere(), project!(M, q))
Optim.project_tangent!(M::NormVort, dq, q) =
    Optim.project_tangent!(Sphere(), project!(M, dq),q)

function relax(rv, Ω, g_tol, mtype, nlz=true)
    cloud = @.cos(π*x/(N+1)/h)*cos(π*y/(N+1)/h) |> Complex
    nlz && normalize!(cloud)
    optimize(
        ψ -> dot(ψ,H(ψ,Ω)) |> real,
        (buf,ψ) -> copyto!(buf, 2*L(ψ,Ω)),
        (z.-rv).*cloud,
        mtype(manifold=NormVort(rv)),
        Optim.Options(iterations=1000, g_tol=g_tol, allow_f_increases=true)
    )
end

function relax2(rv, Ω, g_tol, mtype)
    cloud = @.cos(π*x/(N+1)/h)*cos(π*y/(N+1)/h) |> Complex
    cloud .*= (z.-rv)
    normalize!(cloud)
    optimize(
        ψ -> dot(ψ,H(ψ,Ω)) |> real,
        (buf,ψ) -> copyto!(buf, 2*L(ψ,Ω)),
        cloud,
        mtype(manifold=NormVort(rv)),
        Optim.Options(iterations=1000, g_tol=g_tol, allow_f_increases=true)
    )
end

implot(x,y,image) = plot(x, y, image,
    xlims=(x[1], x[end]), ylims=(y[1], y[end]),
    yflip=false, aspect_ratio=1, framestyle=:box, tick_direction=:out)
implot(image) = implot(x,x,image)
saneportrait(u) = reverse(portrait(u), dims=1)
zplot(u) = implot(saneportrait(u).*abs2.(u)/maximum(abs2,u))
argplot(u) = implot(saneportrait(u))
zplot(u::Matrix{<:Real}) = zplot(u .|> Complex)
argplot(u::Matrix{<:Real}) = argplot(u .|> Complex)
zplot(u::BitArray{2}) = zplot(u .|> Float64)
twoplot(u) = plot(zplot(u), argplot(u))

# Normalizing the cloud, before injecting the vortex, shouldn't help
# relax2(3.0, 0.4, 0.01, ConjugateGradient)
# relax(2.5, 0.4, 0.01, ConjugateGradient)
# relax(2.5, 0.4, 0.01, ConjugateGradient, false)