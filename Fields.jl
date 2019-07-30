module Fields

using FFTW
using ToeplitzMatrices
using LinearAlgebra

import AbstractFFTs.fft
import AbstractFFTs.ifft
import Base: size, getindex, setindex!, similar, BroadcastStyle
import Base.Broadcast: AbstractArrayStyle, Broadcasted
import Base.sum
import Base.diff
import LinearAlgebra.norm

export Field, XField, KField, grid, fft, ifft, sum, diff, norm, lmat

# TODO reinstate parametric eltype, make broadcasting promote the eltype
struct XField <: AbstractArray{Complex{Float64},2}
	h::Tuple{Float64,Float64}
	vals::Array{Complex{Float64},2}
end

struct KField <: AbstractArray{Complex{Float64},2}
	h::Tuple{Float64,Float64}	# this is always the step for the X grid
	vals::Array{Complex{Float64},2}
end

Field = Union{XField, KField}

function XField(h::Tuple{Real, Real}, vals::Array{<:Number,2})
	XField(Tuple{Float64,Float64}(h), Array{Complex{Float64},2}(vals))
end

function KField(h::Tuple{Real, Real}, vals::Array{<:Number,2})
	KField(Tuple{Float64,Float64}(h), Array{Complex{Float64},2}(vals))
end

# AbstractArray primitives

size(U::Field) = size(U.vals)
getindex(U::Field, I...) = getindex(U.vals, I...)
setindex!(U::Field, x, I...) = setindex!(U.vals, x, I...)
function similar(U::F, ::Type{T}, dims::Dims) where {F<:Field, T}
	F(U.h, similar(U.vals, T, dims))
end

# Broadcasting
# TODO reimplement FieldStyle to be generic over XField and KField

struct FieldStyle{F} <: AbstractArrayStyle{2} where F<:Field end
#BroadcastStyle(::Type{F}) where F<:Field = FieldStyle{F}
FieldStyle{F}(::Val{1}) where F = FieldStyle{F}()
FieldStyle{F}(::Val{2}) where F = FieldStyle{F}()
FieldStyle{F}(::Val{N}) where {F,N} = error("Field broadcast with rank 3 array")

BroadcastStyle(::Type{F}) where F<:Field = Broadcast.ArrayStyle{F}()

function similar(bc::Broadcasted{Broadcast.ArrayStyle{F}}, ::Type{T}) where {F,T}
	F(find_h(bc), similar(Array{T}, axes(bc)))
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
promote_h(h, l) = h == l ? h : error("Grids step mismatch in broadcast")

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


# Integrals

sum(u::XField) = prod(u.h)*sum(u.vals)
# TODO remove real when parametric eltypes work again
norm(u::Field) = sqrt(real(sum(abs2.(u))))

# Derivatives

"Finite difference derivative at indexed point"
function diff(I, u::XField, dims...)
	# NYI
	@assert length(dims) == 2
	@assert dims[1] == dims[2]
	# pad periodic boundaries
	vals = u.vals[[end; 1:end; 1], [end; 1:end; 1]]
	I isa CartesianIndex && (I = Tuple(I))
	I = map(x->x+1, I)
	step = Tuple(dims[1] .== 1:2)
	(vals[I.-step...] - 2vals[I...] + vals[I.+step...])/u.h[dims[1]]^2
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