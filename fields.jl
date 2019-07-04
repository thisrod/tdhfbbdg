using FFTW
using Test

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
import PyPlot.contour

struct XField
	h	# 2 element vector with subtype of real
	vals	# figure out how specify a 2D array of a subtype of number
end

struct KField
	h	# this is always the step for the X grid
	vals
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

@testset "fields have the expected grids" begin
	h = 0.2
	U = XField([h, h], ones(3,4))
	X, Y = grid(U);
	
	x = [-0.2, 0, 0.2];
	y = [-0.3, -0.1, 0.1, 0.3]'
	
	@test X.vals ≈ repeat(x, 1, 4)
	@test Y.vals ≈ repeat(y, 3, 1)
end

@testset "integral of cos^2 with single-point axis" begin
	h = π/30
	R = XField([h, 1], ones(30,1))
	x, y = grid(R)
	@test sum(cos(x)^2) ≈ pi/2
end

# TODO set axes to grid and label x, y or kx, ky
function contour(u::Field)
	x, y = grid(u)
	contour(x.vals, y.vals, u.vals)
	gca().set_aspect("equal")
end