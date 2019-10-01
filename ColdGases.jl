module ColdGases

import Fields: grid
using Fields
using Fields: linop_matrix
using LinearAlgebra

export BoseGas, angular_momentum, hamiltonian, hartree_fock

struct BoseGas
	V::XField
	C::Float64
	μ::Float64
	Ω::Float64
end

function BoseGas(; V, C, μ, Ω=0.0)
	BoseGas(V, C, μ, Ω)
end

grid(x::BoseGas) = grid(x.V)

∇²(u) = diff(u,1,1) + diff(u,2,2)
∇²(u,i) = diff(i,u,1,1) + diff(i,u,2,2)

function angular_momentum(s::BoseGas)
	# anonymous functions can have multiple methods
	x, y = grid(s)
	J(ψ::XField) = 1im*(x.*diff(ψ,2) - y.*diff(ψ,1))
	J(ψ::XField,i) = 1im*(x[i]*diff(i,ψ,2) - y[i]*diff(i,ψ,1))
	J
end

function hamiltonian(s::BoseGas)
	H(ψ::XField) = -∇²(ψ) + s.V.*ψ
	H(ψ::XField,i) = -∇²(ψ,i) + s.V[i]*ψ[i]
	H
end

function hartree_fock(s::BoseGas, nc::XField, nnc::XField, sense::Int=1)
	J = angular_momentum(s)
	H = hamiltonian(s)
	L(ψ::XField) = H(ψ) + s.C*(nc+2nnc).*ψ - sense*s.Ω*J(ψ)
	L(ψ::XField,i) = H(ψ,i) + s.C*(nc[i]+2nnc[i])*ψ[i] - sense*s.Ω*J(ψ,i)
	L
end

hartree_fock(s::BoseGas, nc::XField, sense::Int=1) = hartree_fock(s, nc, zero(nc), sense)

function bdg_matrix(s::BoseGas, ψ₀::XField, μ::Real)
	L = hartree_fock(s, 2*abs2.(ψ₀))
	Lstar = hartree_fock(s, 2*abs2.(ψ₀), -1)
	Lmat = linop_matrix(ψ -> L(ψ)-μ*ψ, s.V)
	Lcmat = linop_matrix(ψ -> Lstar(ψ)-μ*ψ, s.V)
	Matrix([Lmat Diagonal(s.C*ψ₀[:].^2); -Diagonal(s.C*conj.(ψ₀[:]).^2) -Lcmat])
end

end