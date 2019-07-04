module FieldsTest

using Fields
using Test

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

end