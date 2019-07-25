module Fields

using FFTW
using ToeplitzMatrices
using LinearAlgebra

import AbstractFFTs.fft
import AbstractFFTs.ifft
import Base.sum
import Base.*
import Base./
import Base.+
import Base.-
import Base.^
import Base.conj
import Base.exp
import Base.sin
import Base.cos
import Base.abs2
import LinearAlgebra.norm

export Field, XField, KField, grid, fft, ifft, sum, norm, apply_fields, apply_field, lmat

struct XField
	h::Tuple{Float64,Float64}
	vals::Array{<:Number,2}
end

struct KField
	h::Tuple{Float64,Float64}	# this is always the step for the X grid
	vals::Array{<:Number,2}
end

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

# Arithmetic on fields
# TODO override broadcasting

Field = Union{XField, KField}

*(u::Field, v::Field) = apply_fields((*), u, v)
+(u::Field, v::Field) = apply_fields((+), u, v)
-(u::Field) = apply_field((-), u)
-(u::Field, v::Field) = apply_fields((-), u, v)
exp(u::Field) = apply_field(exp, u)
abs2(u::Field) = apply_field(abs2, u)
sin(u::Field) = apply_field(sin, u)
cos(u::Field) = apply_field(cos, u)
conj(u::Field) = apply_field(conj, u)

*(c::Number, u::T) where {T <: Field} = T(u.h, c.*u.vals)
*(u::T, c::Number) where {T <: Field} = c*u
/(u::T, c::Number) where {T <: Field} = T(u.h, u.vals./c)
^(u::T, c::Number) where {T <: Field} = T(u.h, u.vals.^c)

# TODO varargs apply_field, deprecate apply_fields
function apply_fields(op, u::T, v::T) where {T <: Field}
	@assert u.h == v.h
	T(u.h, op.(u.vals, v.vals))
end

function apply_field(f, u::T) where {T <: Field}
	T(u.h, f.(u.vals))
end

# Integrals

sum(u::XField) = prod(u.h)*sum(u.vals)
norm(u::Field) = sqrt(sum(abs2(u)))

# Derivatives

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