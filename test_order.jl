module OrderParameterTests

using Fields
using Test

using RK4IP

x, y = begin
	h = 0.1;  l = 15;  n = ceil(Int, l/h)
	R = XField([h, h], ones(n, n))
	grid(R)
end

@testset "lexp solves the 2D heat equation df/dt = ∇²f" begin

	f(t,x) = exp(-x^2/4t)/√t
	f(t,x,y) = f(t,x)*f(t,y)
	
	f₀ = apply_fields((x, y) -> f(0.5, x, y), x, y)
	τ = 0.1
	f₁ = lexp(f₀,τ)
	f₂ = apply_fields((x, y) -> f(0.5 + τ, x, y), x, y)
	
	@test f₁.vals ≈ f₂.vals
end

# Harmonic oscillator and ground state
V = x^2 + y^2
H(ψ) = -∇²(ψ) + V*ψ
φ₀ = exp(-(x^2+y^2)/2)/√π
@assert sum(abs2(φ₀)) ≈ 1

@testset "harmonic oscillator ground state" begin

	Δt = 5e-3

	D(ψ) = lexp(ψ, Δt/2)
	N(ψ) = -Δt*V*ψ
	
	# n-D ground state has energy n
	@test sum(conj(φ₀)*H(φ₀)) ≈ 2
	
	ψ = advance(φ₀,D,N);  ψ /= norm(ψ)
	@test ψ.vals ≈ φ₀.vals
	
	ψ₀ = φ₀*XField([h, h], 1 .+ 0.1*randn(n, n))
	ψ = advance(ψ₀,D,N);  ψ /= norm(ψ)
	@test norm(ψ-φ₀) < norm(ψ-ψ₀)
end

end # module

using Fields
using Test
using RK4IP
using PyPlot

x, y = begin
	h = 0.1;  l = 15;  n = ceil(Int, l/h)
	R = XField([h, h], ones(n, n))
	grid(R)
end

# Harmonic oscillator and ground state
V = x^2 + y^2
H(ψ) = -∇²(ψ) + V*ψ
φ₀ = exp(-(x^2+y^2)/2)/√π

	C = 0.5
	Δt = 1e-2
	
	L(ψ) = H(ψ) + C*abs2(ψ)*ψ
	D(ψ) = lexp(ψ, Δt/2)
	N(ψ) = -Δt*(V + C*abs2(ψ))*ψ
	
	ψ = deepcopy(φ₀)
	μs = zeros(101)
	μs[1] = sum(conj(φ₀)*L(φ₀)).re
	for i = 1:100
		global ψ = advance(ψ, D, N)
		ψ /= norm(ψ)
		global μs[i+1] = sum(conj(ψ)*L(ψ)).re
	end

	clf();  plot(1:101, μs, ".k")

#	@test ψ₁.vals ≈ ψ₀.vals
