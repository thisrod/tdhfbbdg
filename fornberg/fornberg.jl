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

# eigenmodes
index = [1, 3, 6, 10]
Lam, V = eigen(-L)
ii = sortperm(Lam; by=abs)[index]
V = V[:,ii]
Lam = sqrt.(Lam[index]/Lam[1]);

# Plots
rint = r[2:N2+1]
hh = [0;θ]'