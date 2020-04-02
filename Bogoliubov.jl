module Bogoliubov

using Fields
using LinearAlgebra

import Base: length, haskey, get, getindex, setindex!, delete!, iterate, *, /, ==, ≈
export BdGMode, BdGSpectrum, norm2, compact!

# isnothing isn't in the LTS release on OzStar
if VERSION < v"1.1"
	isnothing(::Any) = false
	isnothing(::Nothing) = true
end

struct BdGMode
	u::XField{Complex{Float64}}
	v::XField{Complex{Float64}}
end

*(c::Real, x::BdGMode) = BdGMode(c*x.u, c*x.v)
*(x::BdGMode, c::Real) = c*x
/(x::BdGMode, c::Real) = BdGMode(x.u/c, x.v/c)
==(x::BdGMode, y::BdGMode) = x.u == y.u && x.v == y.v
≈(x::BdGMode, y::BdGMode) = x.u ≈ y.u && x.v ≈ y.v

"""
norm2

In a Minkowski space, squared norms can be negative.
"""
norm2(x::BdGMode) = sum(abs2.(x.u)) - sum(abs2.(x.v))

"""
BdGSpectrum

This acts as a dictionary of (ω=>uv) pairs.

Modes can have degenerate frequencies.  When the push! operation is given the frequency of an existing mode, it jitters it nextfloat.  In contrast, setindex! replaces the mode.

To be implemented: outer constructors BdGSpectrum(::XField) returns an empty Dict, BdGSpectrum(::XField, ::Matrix) computes a spectrum, and the normal Dict constructors.
"""
mutable struct BdGSpectrum <: AbstractDict{Float64, BdGMode}
	# store the ωs and uv matrix, and construct Modes as needed
	# see https://github.com/JuliaLang/julia/issues/25941
	# ew maps frequencies to indices in ev
	# ev is a matrix of column vectors [us; vs]
	grid::XField
	ew::Dict{Float64,Int}
	ev::Matrix{Complex{Float64}}
end

length(x::BdGSpectrum) = length(x.ew)
haskey(x::BdGSpectrum, ω::Float64) = haskey(x.ew, ω)

function getindex(x::BdGSpectrum, ω::Float64)
	i = x.ew[ω]
	BdGMode(
	copyto!(similar(x.grid, Complex{Float64}), x.ev[1:end÷2,i]),
	copyto!(similar(x.grid, Complex{Float64}), x.ev[end÷2+1:end,i]))
end

get(x::BdGSpectrum, ω::Float64, default) = haskey(x,ω) ? x[ω] : default

function setindex!(s::BdGSpectrum, x::BdGMode, ω::Float64)
	if haskey(s,ω)
		s.ev[:,s.ew[ω]] = [x.u[:]; x.v[:]]
	else
		error("No such frequency")
	end
end

# TODO getindex(x, interval) to pick out modes in an energy range

# TODO raise an issue that piggy back iterators are unnecessarily complicated
function iterate(x::BdGSpectrum, s...) 
	ωt = iterate(keys(x.ew), s...)
	isnothing(ωt) && return nothing
	ω, t = ωt
	(ω => x[ω], t)
end

# delete! removes indices, compact! shrinks the ev array
function delete!(x::BdGSpectrum, ω::Float64)
	delete!(x.ew, ω)
	x
end

function compact!(x::BdGSpectrum)
	x.ev = x.ev[:,collect(values(x.ew))]
	k = keys(x.ew)
	x.ew = Dict(zip(k, 1:length(k)))
	x
end

# Constructor from BdG eigenproblem
# TODO exploit Minkowski hermitian property
# TODO warn about complex eigenvalues
function BdGSpectrum(x::XField, L::Matrix{Complex{T}} where T<:Real)
	E = eigen(L)
	D = Dict{Float64}{Int}()
	for i = 1:length(E.values)
		ω = real(E.values[i])
		while haskey(D,ω)
			ω = nextfloat(ω)
		end
		D[ω] = i
	end
	BdGSpectrum(x, D, E.vectors)
end

# standard pairs constructor
function BdGSpectrum(kv)
	# TODO guard type of kv
	G = first(kv).second.u
	S = BdGSpectrum(G,
		Dict{Float64,Int}(),
		Matrix{Complex{Float64}}(undef, 2*length(G), length(kv)))
	for ωuv in kv
		S[ωuv.first] = ωuv.second
	end
end

end