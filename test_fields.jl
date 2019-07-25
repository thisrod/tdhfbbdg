push!(LOAD_PATH, pwd())

module FieldsTest

include("Fields.jl")
using .Fields
using Test

@testset "fields have the expected grids" begin
	h = 0.2
	U = XField((h, h), ones(3,4))
	X, Y = grid(U);
	
	x = [-0.2, 0, 0.2];
	y = [-0.3, -0.1, 0.1, 0.3]'
	
	@test X.vals ≈ repeat(x, 1, 4)
	@test Y.vals ≈ repeat(y, 3, 1)
end

@testset "integral of cos^2 with single-point axis" begin
	h = π/30
	R = XField((h, 1), ones(30,1))
	x, y = grid(R)
	@test sum(cos(x)^2) ≈ pi/2
end

@testset "spectral Laplacian matrix" begin

	# sanity checks
	R = XField((π, 1), ones(2,1))
	@test lmat(R) ≈ [-1 1; 1 -1]/2

	R = XField((π, 1), ones(2,1))
	@test lmat(R) ≈ [-1 1; 1 -1]/2
	
	R = XField((2π/10, 1), ones(10,1))
	x, y = grid(R)
	L = lmat(R)
	@test L*R.vals[:] ≈ zero(R.vals[:]) atol=100*eps()
	
	# odd N execution path on [0,2π]
	Rx = XField((2π/7, 1), ones(7,1))
	x, y = grid(Rx)
	L = lmat(Rx)
	f1 = sin(x)
	@test L*f1.vals[:] ≈ -f1.vals[:]
	
	# even N on scaled domain
	Ry = XField((1, 4π/8), ones(1,8))
	x, y = grid(Ry)
	L = lmat(Ry)
	f2 = sin(y)
	@test L*f2.vals[:] ≈ -f2.vals[:]

	R = XField((2π/4, 2π/5), ones(4,5))
	x, y = grid(R)
	L = lmat(R)
	fx = sin(x)
	@test L*fx.vals[:] ≈ -fx.vals[:]
	fy = cos(y)
	@test L*fy.vals[:] ≈ -fy.vals[:]
	fxy = sin(2x+y)
	@test L*fxy.vals[:] ≈ -5fxy.vals[:]
end

end