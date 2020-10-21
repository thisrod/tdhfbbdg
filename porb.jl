# Investigate contstrained vortex positions

using LinearAlgebra, Plots, Optim, Tables
using Revise
using Superfluids
using Superfluids: relax, winding, loopixs, find_vortices, poles, cluster_adjacent, adjacent_index, cloud

default(:legend, :none)
Superfluids.default!(:xlims, (-6,6))
Superfluids.default!(:ylims, (-6,6))

s = Superfluid{2}(500, (x,y)->(x^2+y^2)/2)
# d = FourierDiscretisation{2}(66, 0.3)
d = FourierDiscretisation{2}(200, 20/199)
g_tol = 1e-7
Ω = 0.3

L, H, J = Superfluids.operators(s,d,:L,:H,:J)

ψ = steady_state(s,d)
μ = dot(L(ψ), ψ) |> real
E₀ = dot(H(ψ), ψ) |> real
R_TF = sqrt(2μ)

rr = range(0.0, R_TF, length=10)
rr = rr[2:end]
lixs = loopixs(d, 1.1R_TF)

ix = argmin(@. abs(z-rr[end-1]))
r = z[ix]

j = 1
result = relax(s, d; rvs=Complex{Float64}[r], Ω, g_tol, iterations=j)
w = result.minimizer
result = relax(s, d; initial=cloud(d,r), Ω, g_tol, iterations=j)
q = result.minimizer

results = []
for r = rr
    result = relax(s, d; rvs=Complex{Float64}[r], Ω, g_tol, iterations=5000, a=0.2)
    q = result.minimizer
    w, μ = [J(q)[:] q[:]] \ L(q)[:] |> real
    rd = L(q;Ω=w)-μ*q
    sd = copy(rd)
    M = PinnedVortices(d, r)
    sd[M.ixs[:]] .= 0
    push!(results, (
        r = r,
        itns = result.iterations,
        cflag = Optim.converged(result),
        q = q,
        winding = winding(q, lixs),
        w = w,
        rdl = sum(abs2, rd),
        sdl = sum(abs2, sd)
    ))
end
println("Done")
results = Tables.columntable(results)

ww = range(0.2, 0.3, length=6)
qs = []
for Ω = ww
    result = relax(s, d; rvs=Complex{Float64}[(rr[end-1]+rr[end])/2], Ω, g_tol, iterations=5000)
    push!(qs, result.minimizer)
end 

uu = exp.(2π*1im*(0:0.01:1))

j = 8
r = results.r[j]
q = results.q(j)
z = Superfluids.argand(d)
M = PinnedVortices(d, r)
ixs = M.ixs[:]
jxs = @. abs(z-r) < 5d.h
u = zero(z)
u[ixs] = normalize(z[ixs].-r)

A = [z[ixs].-r conj(z[ixs]).-r ones(size(z[ixs]))]
Q, R = qr(A)
cs = Q'*(z.-r)[ixs]


for j = Tables.rows(results)
    plot(d, @. j.q/abs(j.q))
    scatter!([j.r], [0])
    plot!(real(1.1R_TF*uu), imag(1.1R_TF*uu), lc=:white) |> display
    sleep(2)
end

plot(scatter(results.r, results.rdl), scatter(results.r, results.w), layout=@layout[a;b])

# secant method to find Ω fixed point
function woff(r, a)
    result = relax(s, d; rvs=Complex{Float64}[r], Ω, g_tol, iterations=5000, a)
    q = result.minimizer
    w, μ = [J(q)[:] q[:]] \ L(q)[:] |> real
    w - Ω
end

r1 = 0.0
s1 = woff(r1, 0.2)
r2 = R_TF
s2 = woff(r2, 0.2)

for j = 1:10
    r3 = (r1*s2-r2*s1)/(s2-s1)
    r1 = r2
    s1 = s2
    r2 = r3
    s2 = woff(r2, 0.2)
end

plotly()

plot(d,u,xlims=(r-5d.h, r+5d.h), ylims=(-5d.h, 5d.h));
scatter!([r], [0]);
title!("L(q) with phase relative to q")

v = cloud(d,r)
ll = L(v, Ω)
dl = @. ll / (v/abs(v))
    
function show_step(q, r)
    ll = L(q;Ω)
    ll ./= (q./abs.(q))
    plot(p(real(ll), r), p(imag(ll), r))
end

function p(q, r, n = 3)
    jxs = @. abs(z-r) < 2n*d.h
    u = zero(z)
    u[jxs] = q[jxs]
    plot(d,u,xlims=(r-n*d.h, r+n*d.h), ylims=(-n*d.h, n*d.h))
    scatter!([r], [0])
end

# Pairs: everything appears to have vortices
# 200×200 grid converges in 300 iterations, 66×66 takes 2000 for pairs
# Residual has opposite sign on 4 pixels and surrounds
# Disperse residual comparable at different grid sizes
# 4 pixel residual increases with large grids
# (Can the residual tell us where to move the vortices?)