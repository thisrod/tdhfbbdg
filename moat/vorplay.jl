# Playing with Wannier derivatives and vortex detection

using JLD2
@load "moa.jld2"
include("../system.jl")
include("../figs.jl")

q = Su[1]
P, Q = poles(q)
A = @. (r<5)*P/q;  B = @. (r<5)*Q/q

# Changing magnitude has P=Q*, changing phase P=-Q*
# abs(P+Q*)/abs(P-Q*) picks up moat "vortex" 

# (A+B) should be magnitude, real(A-B) should be phase
# plot(argplot(q), zplot(real.(A+B)), zplot(real.(A-B)))

Dabs = @. (A+conj(B)) / 2
Dang = @. (A-conj(B)) / 2
C = (abs.(P) + abs.(Q))/2

hm(u) = heatmap(y,y,abs.(u),aspect_ratio=1)
function mp(u)
    zplot(u)
    scatter!([-R], [0], ms=1, mc=:white, leg=:none)
end
function mp(u, a)
    mp(u)
    xlims!(-R-a,-R+a)
    ylims!(-a,+a)
end

A = @. (r<5)*P/q;  B = @. (r<5)*Q/q;

a = 1.0
P1 = argplot(q)
xlims!(-R-a,-R+a)
ylims!(-a,+a)
plot(P1, mp(A, a), mp(B, a))

function wplot(u, v)
    P, Q = poles(u)
    P1 = zplot(P)
    title!("$(norm(P, Inf))")
    P2 = zplot(-conj(Q))
    title!("$(norm(Q, Inf))")
    plot(zplot(u), P1, P2, zplot(v))
end

# x = exp(1im*2π*rand()); wplot(real(x*z), x*ones(size(z)))
# x = exp(1im*2π*rand()); wplot(1im*real(x*z), x*ones(size(z)))