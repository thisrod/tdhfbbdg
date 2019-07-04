module RK4IP

using Fields

export lap, explap, nlptl, itime_step

# D operator from Appendix C of Caradoc-Davies
# All this is in imaginary time

function lap(u)
	v = fft(u)
	kx, ky = grid(v)
	ifft(-(kx^2+ky^2) * v)
end

# exponential Laplacian
function explap(u, τ)
	v = fft(u)
	kx, ky = grid(v)
	ifft(exp(-τ*(kx^2+ky^2)) * v)
end

# N = V + C|ψ|².  Time independent for now.

function nlptl(u, V, C)
	x, y = grid(u)
	apply_fields(V, x, y)*u + C*abs2(u)
end

# Algorithm B.10 of Caradoc-Davies
function itime_step(ψ, D, N)
	ψ_I = D(ψ)
	k₁ = D(N(ψ))
	k₂ = N(ψ_I + k₁/2)
	k₃ = N(ψ_I + k₂/2)
	k₄ = N(D(ψ_I + k₃))
	@assert 0.5*norm(ψ) > norm(k₁) + norm(k₂) + norm(k₃) + norm(k₄) 
	D(ψ_I + (k₁ + 2(k₂+k₃))/6) + k₄/6
end
	
end