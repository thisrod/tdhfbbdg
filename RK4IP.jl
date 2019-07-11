module RK4IP

using Fields
using LinearAlgebra

export ∇², lexp, nlptl, advance, gauss_seidel_step, adapt_relax, ar_set

function ∇²(u)
	v = fft(u)
	kx, ky = grid(v)
	ifft(-(kx^2+ky^2) * v)
end

# exponential Laplacian.  Returns exp(τ∇²) u
function lexp(u, τ)
	v = fft(u)
	kx, ky = grid(v)
	ifft(exp(-τ*(kx^2+ky^2)) * v)
end

# Algorithm B.10 of Caradoc-Davies
function advance(ψ, D, N)
	ψ_I = D(ψ)
	k₁ = D(N(ψ))
	k₂ = N(ψ_I + k₁/2)
	k₃ = N(ψ_I + k₂/2)
	k₄ = N(D(ψ_I + k₃))
	@assert 0.5*norm(ψ) > norm(k₁) + norm(k₂) + norm(k₃) + norm(k₄) 
	D(ψ_I + (k₁ + 2(k₂+k₃))/6) + k₄/6
end

ar_gap = 15
ar_factor = 2

function ar_set(gap, factor)
	global ar_gap, ar_factor
	ar_gap = gap
	ar_factor = factor
	nothing
end

"""
    adapt_relax(step, x₀, a₀, R)

The function |step(x, a)| takes an estimate |x| and an aggressiveness |a|, which determines how hard it will try to improve it.  For example, |a| could be the time step in imaginary time propagation or the extrapolation parameter in successive over-relaxation.  The function returns an improved estimate |x'| and a residual |r|.  It is called iteratively until the residual reduces to |R|.  Typically, |a=0| will do nothing, and overly aggressive optimisation will blow up.  The parameter |a| is adjusted to maximise the rate of reduction in |R|.
"""
function adapt_relax(step, x₀, a₀, R)
	# TODO improve guess a₀ by trying a₀/2, a₀ and 2a₀, fitting
	# a parabola, finding the minimum R, and keeping the lowest
	# 3 guesses plus the two to either side of them.  That should
	# be a helper function.
	
	# TODO find a way to keep track of μ as well as the residual
	
	# TODO adjust a continuously by extrapolation
	
	global relaxed_a, relaxed_r
	x = x₀; a = a₀; r = Inf
	relaxed_a = Float64[a₀]
	relaxed_r = Float64[step(x₀,a₀)[2]]
	for i = 1:div(1000,ar_gap)
		for j = 1:ar_gap
			x, r₁ = step(x,a)
			r₁ < r || break
			r₁ > R || return x
			r = r₁
		end
		# TODO find out if Julia has has tables
		trials = [a/ar_factor, a, a*ar_factor]
		rs = [step(x,b)[2] for b = trials]
		a = trials[argmin(rs)]
		push!(relaxed_a, a)
		push!(relaxed_r, r)
	end
	println("WARNING: adapt_relax failed to converge")
	x
end

"""
    gauss_seidel_step(A, b, y₀)

Return y₁, the vector y₀ advanced by one step of the Gauss-Seigel algorithm.

Extra method.  A is a function.  A(y, i) returns A[i,:]⋅y, and A(i) returns 
"""
function gauss_seidel_step(A, b, y₀)
	y = copy(y₀)
	for i = eachindex(y)
		y[i] = 0
		y[i] = (b[i]-A[i,:]⋅y)/A[i,i]
	end
	y
end

end