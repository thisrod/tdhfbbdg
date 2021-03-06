module Fields

using FFTW
using ToeplitzMatrices
using LinearAlgebra

import AbstractFFTs.fft
import AbstractFFTs.ifft
import Base: size, getindex, setindex!, similar, BroadcastStyle
import Base.Broadcast: AbstractArrayStyle, Broadcasted
import Base: sum, diff, ==, ≠, *, /, \, show
import LinearAlgebra: norm, Matrix

using DomainSets: (..), ×

export Field, XField, KField, grid, fft, ifft, sum, diff, norm, lmat, braket,
	domain

struct XField{T<:Number} <: AbstractMatrix{T}
	h::Tuple{Float64,Float64}
	vals::Matrix{T}
end

fieldtypeof(u::XField) = XField
fieldtypeof(::Type{XField{T}}) where T = XField

struct KField{T<:Number} <: AbstractMatrix{T}
	h::Tuple{Float64,Float64}	# this is always the step for the X grid
	vals::Matrix{T}
end

fieldtypeof(u::KField) = KField
fieldtypeof(::Type{KField{T}}) where T = KField

Field{T} = Union{XField{T}, KField{T}} where T

function XField(h::Tuple{Real, Real}, vals::Matrix)
	XField(Tuple{Float64,Float64}(h), vals)
end
XField(h::Real, vals::Matrix) = XField((h,h), vals)

function KField(h::Tuple{Real, Real}, vals::Matrix)
	KField(Tuple{Float64,Float64}(h), vals)
end
KField(h::Real, vals::Matrix) = KField((h,h), vals)

# Display

function domain(u::XField)
	hs = u.h;  ns = size(u.vals)
	interval(h,n) = h/2*(1-n..n-1)
	interval(hs[1],ns[1])×interval(hs[2],ns[2])
end

show(io::IO, u::XField{T}) where T = print(io, "XField{", T, "} on ", domain(u))

# AbstractArray primitives

size(U::Field) = size(U.vals)
getindex(U::Field, I...) = getindex(U.vals, I...)
setindex!(U::Field, x, I...) = setindex!(U.vals, x, I...)
function similar(U::Field, ::Type{T}, dims::Dims) where T
	fieldtypeof(U){T}(U.h, similar(U.vals, T, dims))
end

# TODO open issue where u == v does something special, but u ≈ v broadcasts normally

# Here's the workaround

==(u::Field, v::Field) = all(u .== v)
≠(u::Field, v::Field) = !(u == v)

# Catch errors

*(u::Field, v::Field) = throw(ArgumentError("matmul with Fields, try .* instead of *"))
/(u::Field, v::Field) = throw(ArgumentError("matmul with Fields, try ./ instead of /"))
\(u::Field, v::Field) = throw(ArgumentError("matmul with Fields, try .\\ instead of \\"))

# Broadcasting

struct FieldStyle{F} <: AbstractArrayStyle{2} where F<:Field end
#BroadcastStyle(::Type{F}) where F<:Field = FieldStyle{F}
FieldStyle{F}(::Val{1}) where F = FieldStyle{F}()
FieldStyle{F}(::Val{2}) where F = FieldStyle{F}()
FieldStyle{F}(::Val{N}) where {F,N} = error("Field broadcast with rank 3 array")

BroadcastStyle(::Type{F}) where F<:Field = Broadcast.ArrayStyle{fieldtypeof(F)}()

function similar(bc::Broadcasted{Broadcast.ArrayStyle{F}}, ::Type{T}) where {F<:Field,T}
	F{T}(find_h(bc), similar(Array{T}, axes(bc)))
end

find_h(x) = nothing
find_h(bc::Base.Broadcast.Broadcasted) = find_h(bc.args)
function find_h(args::Tuple)
	if isempty(args)
		nothing
	else
		promote_h(find_h(args[1]), find_h(Base.tail(args)))
	end
end
find_h(U::Field) = U.h

promote_h(::Nothing, ::Nothing) = nothing
promote_h(h, ::Nothing) = h
promote_h(::Nothing, h) = h
# TODO find an alternative to comparing floating point grids for exact equality
promote_h(h, l) = h == l ? h : throw(DimensionMismatch("grids of Fields must match"))

# Fourier transforms
# TODO make these L² unitary

function fft(u::XField)
	KField(u.h, fft(u.vals))
end

function ifft(u::KField)
	XField(u.h, ifft(u.vals))
end

# grid returns two fields, with the two coordinates.
function grid(u::XField)
	h = u.h;  n = size(u.vals)
	X(h,n) = [h*(m - n/2 + 1/2) for m=0:n-1]
	x = XField(h, repeat(X(h[1], n[1]), 1, n[2]))
	y = XField(h, repeat(X(h[2], n[2])', n[1], 1))
	x, y
end

function grid(u::KField)
	h = u.h;  n = size(u.vals)
	K(h,n) = 2π/h/n*[m ≤ n//2 ? m : m-n for m=0:n-1]
	kx = KField(h, repeat(K(h[1], n[1]), 1, n[2]))
	ky = KField(h, repeat(K(h[2], n[2])', n[1], 1))
	kx, ky
end

# matrices for linear operators
# TODO make derivative operators a callable object, with a constructor Matrix(::LinOp)

braket(f, ψ::Field) = sum(conj.(ψ).*f(ψ))

function linop_matrix(D, F::XField)
	# The derivative of e_i goes in the ith column of the matrix
	A = fill(Complex(NaN), length(F), length(F))
	ei = zero(F)
	j = 1
	for i = eachindex(ei)
		ei[i] = 1
		A[:,j] = D(ei).vals[:]
		ei[i] = 0
		j += 1
	end
	A
end


# Integrals

sum(u::XField) = prod(u.h)*sum(u.vals)
# TODO remove real when parametric eltypes work again
norm(u::Field) = sqrt(real(sum(abs2.(u))))

# Derivatives

"Finite difference derivative at indexed point"
function diff(I, u::XField, dims...)
	# stencil = [1, -2, 1]
	# stencil = [-1/12, 4/3, -5/2, 4/3, -1/12]
	stencil = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
	n = length(dims) == 2 ? length(stencil) : 3
	# Implementation restrictions
	@assert length(dims) ≤ 2
	@assert all(j->j==dims[1], dims)
	# pad periodic boundaries
	# vals = u.vals[[end; 1:end; 1], [end; 1:end; 1]]
	# set zero boundaries
	m = (n-1)÷2	# margin width
	vals = zeros(eltype(u), size(u).+2m)
	vals[1+m:end-m, 1+m:end-m] = u.vals
	I isa CartesianIndex && (I = Tuple(I))
	I = map(x->x+m, I)
	# mask to increment index along the differentiation axis
	step = Tuple(dims[1] .== 1:2)
	if length(dims) == 2
		sum(stencil[j]*vals[I .+ (j-(n+1)÷2).*step...] for j = 1:n)/u.h[dims[1]]^2
	else
		(vals[I.+step...] - vals[I.-step...])/2u.h[dims[1]]
	end
end

function diff(u::XField, dims...)
	du = similar(u)
	for i = eachindex(u)
		du[i] = diff(i, u, dims...)
	end
	du
end

"Spectral Laplacian matrix"
function lmat(u::XField)
	n1 = size(u.vals,1);  n2 = size(u.vals,2)
	D1 = kron(Matrix(I,n2,n2), dmat1(u.h[1], n1))
	D2 = kron(dmat1(u.h[2], n2), Matrix(I,n1,n1))
	# show("D1 = $D1  D2 = $D2")
	D1 + D2
end

function dmat1(h::Real, n::Integer)
	j = 0:n-1
	# formulae for domain [0, 2π] with integer wave numbers
	if n == 1
		return fill(0.,1,1)
	elseif iseven(n)
		rc = -1/2*(-1).^j.*csc.(j*π/n).^2
		rc[1] = -n^2/12 - 1/6
	else
		rc = -1/2*(-1).^j.*csc.(j*π/n).*cot.(j*π/n)
		rc[1] = -n^2/12 + 1/12
	end
	# show("n = $n h = $h rc = $rc")
	# scale to [0, n*h]
	rc *= (2π/n/h)^2
	# show("scaled rc = $rc")
	Toeplitz(rc,rc)
end

end