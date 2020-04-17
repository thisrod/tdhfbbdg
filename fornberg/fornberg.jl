# Julia port of Program 28 from SMM

using LinearAlgebra, ToeplitzMatrices

function cheb(N::Integer)
    N == 0 && return (0, 1)
    x = cos.(π*(0:N)/N)
    c = [2; ones(N-1,1); 2].*(-1).^(0:N)
    dX = x.-x'
    D  = c ./ c' ./ (dX+I)					# off-diagonal entries
    D  -= sum(D; dims=2)[:] |> Diagonal		# diagonal entries
    D, x
end

# radial derivatives
N = 25;  @assert N % 2 == 1
N2 = (N-1)÷2
D, r = cheb(N)
D2 = D^2
D1 = D2[2:N2+1,2:N2+1]; D2 = D2[2:N2+1,N:-1:N2+2]
E1 =  D[2:N2+1,2:N2+1]; E2 =  D[2:N2+1,N:-1:N2+2]

# circular derivatives
M = 20;   @assert M % 2 == 0
dθ = 2π/M;  θ = dθ*(1:M); M2 = M÷2
rc = [-π^2/(3*dθ^2)-1/6;  0.5*(-1).^(2:M)./sin.(dθ*(1:M-1)/2).^2]
D2θ = Toeplitz(rc, rc)

# Laplacian in polar coordinates:
rint = r[2:N2+1]
R = 1 ./ rint |> Diagonal
Z = zeros(M2,M2)
L = kron(D1+R*E1,Matrix(I,M,M)) +
    kron(D2+R*E2,[Z I;I Z]) +
    kron(R^2,D2θ)

# U = rint'.*exp.(-5rint'.^2 .+ 1im*θ);

V = 5*kron(Diagonal(rint.^2),Matrix(I,M,M))

# eigenmodes
index = [1, 3, 6, 10]
Lam, ev = eigen(-L.+V)
ii = sortperm(Lam; by=abs)[index]
ev = ev[:,ii]
Lam = sqrt.(Lam[index]/Lam[1]);

mode(j) = reshape(ev[:,j], M, N2)

# Plots
rr = (rint[1:end-1] .+ rint[2:end])/2
rr = [1; rr; 0]
hh = (θ[1:end-1] .+ θ[2:end])/2
hh = [dθ/2;hh;θ[end]+dθ/2]

plr((r, h)) = (r*cos(h), r*sin(h))
rh(j,k) = [(rr[j], hh[k]), (rr[j+1], hh[k]), (rr[j+1], hh[k+1]), (rr[j], hh[k+1]), (rr[j], hh[k])]
vtxs(j,k) = plr.(rh(j,k))
ppix(j,k) = vtxs(j,k) |> Shape

function rplot(u)
    P = plot(
        bg = :black,
        xlim = (-1, 1),
        ylim = (-1, 1),
        framestyle = :none,
        legend = false,
        aspect_ratio = 1
    )
    clrs = portrait(u).*abs2.(u)/maximum(abs2.(u))
    for j = 1:length(rr)-1
        for k = 1:length(hh)-1
            plot!(P, ppix(j,k), fillcolor = clrs[k,j], lw=0, leg=:none)
        end
    end
    display(P)
end

mode(j) = reshape(ev[:,j], M, N2)