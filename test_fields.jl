push!(LOAD_PATH, pwd())

module FieldsTest

include("Fields.jl")
using .Fields
using .Fields: linop_matrix
using Test

@testset "fields have the expected grids" begin
	h = 0.2
	U = XField((h, h), ones(3,4))
	V = XField(h, ones(3,4))
	X, Y = grid(U);
	
	x = [-0.2, 0, 0.2];
	y = [-0.3, -0.1, 0.1, 0.3]'
	
	@test U.h == V.h
	@test U.vals == V.vals
	@test X.vals ≈ repeat(x, 1, 4)
	@test Y.vals ≈ repeat(y, 3, 1)
end

@testset "Field subtypes" begin
	X = XField((0.1, 1), ones(Int,3,1))
	Y = fft(X)
	
	@test X isa XField{Int}
	@test X isa XField
	@test X isa Field{Int}
	@test X isa Field
	@test Y isa KField{<:Complex}
	@test Y isa Field{eltype(Y)}
	@test Y isa Field
end

@testset "type stability in broadcasting" begin
	h = 0.2
	X = XField((h, h), ones(Int,3,4))
	Y = XField((h, h), ones(Float64,3,4))
	
	@test X isa XField{Int}
	@test Y isa XField{Float64}
	@test copy(X) isa typeof(X)
	@test X./2 isa XField{<:Real}
	@test cos.(X) isa XField{<:Real}
	@test cos.(X).^2 isa XField{<:Real}
	@test X+X isa typeof(X)
	@test Y+Y isa typeof(Y)
	@test X.+Y isa typeof(Y)
	@test X+Y isa typeof(Y)
	
	χ = fft(Y)
	@test χ isa KField{Complex{Float64}}
	@test copy(χ) isa typeof(χ)
	@test cos.(χ) isa typeof(χ)
	@test χ.+fft(X) isa KField
	@test χ+fft(X) isa KField
	@test cos.(χ).^2 isa typeof(χ)
	@test abs2.(χ) isa KField{<:Real}
end

@testset "integral of cos^2 with single-point axis" begin
	h = π/30
	R = XField((h, 1), ones(30,1))
	x, y = grid(R)
	@test sum(cos.(x).^2) ≈ π/2
	@test norm(cos.(x)) ≈ sqrt(π/2)
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
	f1 = sin.(x)
	@test L*f1.vals[:] ≈ -f1.vals[:]
	
	# even N on scaled domain
	Ry = XField((1, 4π/8), ones(1,8))
	x, y = grid(Ry)
	L = lmat(Ry)
	f2 = sin.(y)
	@test L*f2.vals[:] ≈ -f2.vals[:]

	R = XField((2π/4, 2π/5), ones(4,5))
	x, y = grid(R)
	L = lmat(R)
	fx = sin.(x)
	@test L*fx.vals[:] ≈ -fx.vals[:]
	fy = cos.(y)
	@test L*fy.vals[:] ≈ -fy.vals[:]
	fxy = sin.(2x+y)
	@test L*fxy.vals[:] ≈ -5fxy.vals[:]
end

@testset "finite difference derivatives" begin

	# TODO test boundary

	R = XField((0.1,1), ones(51,1));
	x,y = grid(R)
	@test diff((26,1), x.^2, 1) ≈ 0
	@test diff((36,1), x.^2, 1) ≈ 2x[36,1]
	@test diff((26,1), x.^2, 1, 1) ≈ 2
	@test diff(x.^2, 1, 1)[26,1] == diff((26,1), x.^2, 1, 1)

end

@testset "derivative matrices" begin
	R = XField((0.2, π), zeros(3,5))
	
	function ∇²(u)
		v = fft(u)
		kx, ky = grid(v)
		ifft(-(kx.^2+ky.^2) .* v)
	end
	
	@test lmat(R) ≈ linop_matrix(∇², R)
end

end
