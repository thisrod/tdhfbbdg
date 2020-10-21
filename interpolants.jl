# Plots of odd circular interpolants and spectral derivatives

N = 11;  h = 2π/N
p(x) = (h/2π)*sin(N*x/2)/sin(x/2)
dp(x) = (cos(N*x/2)*sin(x/2)-sin(N*x/2)*cos(x/2)/N)/(2*sin(x/2)^2)
ddp(j) = (-1)^(j+1)*cos(j*h/2)/2/sin(j*h/2)^2
jj = -(N-1)/2:(3N-1)/2
x = h.*jj
pp = p.(x);  pp[mod.(x,2π).≈0] .= 1
dd = ddp.(jj);  dd[mod.(x,2π).≈0] .= 1/12 - π^2/3h^2


# layer(p, -π, 3π)
plot(
	scatter(x, pp),
	plot(dp, -π, 3π),
	plot(x=xx, y=dd)
)
