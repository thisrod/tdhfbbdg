module Bogoliubov

using Fields

import Base: size, getindex, setindex!, similar, BroadcastStyle
export BdGMode, BdGSpectrum

struct BdGMode
	ω::Float64
	u::XField
	v::XField
end

struct BdGSpectrum <: AbstractVector{BdGMode}
	# store the ωs and uv matrix, and construct Modes as needed
	grid::XField
	ew::Vector{Float64}
	ev::Matrix
end

size(x::BdGSpectrum) = size(x.ew)

function getindex(x::BdGSpectrum, i)
	BdGMode(
	x.ew[i],
	copyto!(similar(x.grid), x.ev[1:end÷2,i]),
	copyto!(similar(x.grid), x.ev[end÷2+1:end,i]))
end

end