# Find steady rotating lattices

using Optim, Plots
using Superfluids
using Superfluids: unroll, relax
using LinearAlgebra: dot

s = Superfluid{2}(100.0, (x,y)->(x^2+y^2)/2) |> Superfluids.default!
# d = FourierDiscretisation{2}(100, 0.121) |> Superfluids.default!
d = FourierDiscretisation{2}(100, 0.150) |> Superfluids.default!
L, J = Superfluids.operators(:L,:J)

g_tol = 1e-6

ψ = steady_state()
μ = dot(L(ψ), ψ) |> real
RTF = sqrt(2μ)

ixs = 
    begin
        rix = 4.5
        z = argand(d)
        nix = floor(2π*rix/d.h)
        ws = @. rix*exp(2π*1im*(0:nix)/nix)
        ixs = [argmin(@. abs(z-w)) for w in ws]
    end

function winding(q, r=RTF)
   z = argand(d)
   nix = floor(2π*r/d.h)
   ws = @. r*exp(2π*im*(0:nix)/nix)
   ixs = [argmin(@. abs(z-w)) for w in ws]
   hh = unroll(angle.(q[ixs]))/2π
   round(Int, hh[end]-hh[1])
end

btol = 0.01

# Bisection search to find correct winding number, then Optim to match rotating frame
function bracket(rvs, a, l=0.0, h=1.0, rout=2RTF)

    global stack
    stack = []
    
    as = fill(a,length(rvs))
    q = similar(ψ)
    rin = 1.1*maximum(abs, rvs)
    @info rin rout
    
    function bkts(Ω, w1, w2, w3, w4)
        @info "Bracketing" (w1, w2, w3, w4)
        woff = wdisc!(Ω, 1000)
        push!(stack, (Ω, copy(q)))
        @info "Windings" winding(q, rin) winding(q, rout)
        ws =
            if winding(q, rin) < length(rvs)
                [Ω, w2, w3, w4]
            elseif winding(q, rout) > length(rvs)
                [w1, w2, w3, Ω]
            elseif woff < 0
                [w1, Ω, w3, w4]
            elseif woff > 0
                [w1, w2, Ω, w4]
            else
                @assert false
            end
        @assert issorted(skipmissing(ws))
        ws
    end
    
    function wdisc!(Ω, iterations)
        result = relax(s, d; rvs, Ω, g_tol, iterations, as)
        q .= result.minimizer
        W, μ = [J(q)[:] q[:]] \ L(q)[:] |> real
        W-Ω
    end
    
    gap(l::Real, ::Missing, ::Missing, h::Real) = l, h
    gap(_, l::Real, ::Missing, h::Real) = l, h
    gap(l::Real, ::Missing, h::Real, _) = l, h

    function bkt(w1, w2, w3, w4)
        l, h = gap(w1, w2, w3, w4)
        h - l < btol && error("Squeeze ", [w1, w2, w3, w4])
        Ω = (l+h)/2
        bkt(bkts(Ω, w1, w2, w3, w4)...)
   end
    
    function bkt(_, l::Real, r::Real, _)
        @info "Relaxing" (l, r)
        result = optimize(w->abs2(wdisc!(w, 5000)), l, r, abs_tol=g_tol)
        result.minimizer, q
    end
    
    bs = [l, missing, missing, h]
    bs = bkts(l, bs...)
    bs = bkts(h, bs...)
    bkt(bs...)
end

# u = complex.([-1.0, 0, 1]);  qs = [];  Ws = [];  for r = (1:5)/5*RTF  push!(Ws, bracket(r*u, 0.3));  push!(qs, copy(q)) end